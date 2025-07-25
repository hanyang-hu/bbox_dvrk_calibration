import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
    unscented_mix_angle_to_axis_angle
)

import math
import torch
import kornia
import nvdiffrast.torch as dr
from pytorch3d.renderer import MeshRasterizer
import numpy as np
import cv2
import argparse
import time
import warnings


def smallest_diag_dominating_A_cvxpy(A: np.ndarray) -> np.ndarray:
    import cvxpy
    """
    Given symmetric matrix A (3x3),
    find diagonal D = diag(d) minimizing sum(d)
    s.t. D - A is positive semidefinite.

    Returns:
        D: smallest diagonal matrix dominating A
    """
    assert A.shape == (3, 3)
    assert np.allclose(A, A.T, atol=1e-8), "A must be symmetric"

    d = cp.Variable(3, nonneg=True)  # diagonal entries

    D = cp.diag(d)
    constraints = [D - A >> 0]  # PSD constraint

    prob = cp.Problem(cp.Minimize(cp.sum(d)), constraints)
    prob.solve(solver=cp.SCS)  # or cp.MOSEK if available

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Solver did not find an optimal solution")

    D_opt = np.diag(d.value)
    return D_opt


def parseArgs():
    parser = argparse.ArgumentParser()
    data_dir = "data/consistency_evaluation/easy/4"
    parser.add_argument("--data_dir", type=str, default=data_dir)  # reference mask
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument(
        "--joint_file", type=str, default=os.path.join(data_dir, "joint_0183.npy")
    )  # joint angles
    parser.add_argument(
        "--jaw_file", type=str, default=os.path.join(data_dir, "jaw_0183.npy")
    )  # jaw angles
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=30)
    
    args = parser.parse_args()

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707
    args.scale = 1.0

    # clip space parameters
    args.znear = 0
    args.zfar = float('inf')

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    args.use_nvdiffrast = False # do not use nvdiffrast in CtRNet

    return args


def transform_mesh(cameras, mesh, R, T, args):
    """
    Transform the mesh from world space to clip space
    Modified from https://github.com/NVlabs/nvdiffrast/issues/148#issuecomment-2090054967
    """
    # world to view transform
    verts = mesh.verts_padded()  #  (B, N_v, 3)
    verts_view = cameras.get_world_to_view_transform(R=R, T=T).transform_points(verts)  # (B, N_v, 3)
    verts_view[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
    verts_view_home = torch.cat([verts_view, torch.ones_like(verts_view[..., [0]])], axis=-1) # (B, N_v, 4)

    # projection
    fx, fy = cameras.focal_length[0]
    px, py = cameras.principal_point[0]
    height, width = cameras.image_size[0]
    near, far = args.znear, args.zfar
    A = (2 * fx) / width
    B = (2 * fy) / height
    C = (width - 2 * px) / width
    D = (height - 2 * py) / height
    E = (near + far) / (near - far)
    F = (2 * near * far) / (near - far)
    t_mtx = projectionMatrix = torch.tensor(
        [
            [A, 0, C, 0],
            [0, B, D, 0],
            [0, 0, E, F],
            [0, 0, -1, 0]
        ]
    ).to(verts.device)
    verts_clip = torch.matmul(verts_view_home, t_mtx.transpose(0, 1))

    faces_clip = mesh.faces_padded().to(torch.int32)

    return verts_clip, faces_clip


def render(glctx, pos, pos_idx, resolution: [int, int], antialiasing=False):
    """
    Silhouette rendering pipeline based on NvDiffRast
    """
    # Create color attributes
    col = torch.ones_like(pos[..., :1], dtype=torch.float32) # (B, N_v, 1)
    col_idx = pos_idx

    # Render the mesh
    rast_out, _ = dr.rasterize(glctx, pos, pos_idx, resolution=resolution, grad_db=False)
    color   , _ = dr.interpolate(col, rast_out, col_idx)
    if antialiasing:
        color = dr.antialias(color, rast_out, pos, pos_idx)
    return color.squeeze(-1) # (B, H, W)


if __name__ == "__main__":
    with torch.no_grad():
        warnings.filterwarnings("ignore", category=UserWarning)
        # torch.manual_seed(42)
        # np.random.seed(42)

        args = parseArgs()
        
        joints = np.load(args.joint_file)
        jaw = np.load(args.jaw_file)

        """Or just for a single image processing"""

        # a.1) Build model
        model = CtRNet(args)
        mesh_files = [
            f"{args.mesh_dir}/shaft_multi_cylinder.ply",
            f"{args.mesh_dir}/logo_low_res_1.ply",
            f"{args.mesh_dir}/jawright_lowres.ply",
            f"{args.mesh_dir}/jawleft_lowres.ply",
        ]

        robot_renderer = model.setup_robot_renderer(mesh_files)
        robot_renderer.set_mesh_visibility([True, True, True, True])

        # a.2) Joint angles (same for all items, or replicate if needed)
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = torch.tensor(
            joint_angles_np, device=model.device, requires_grad=False, dtype=torch.float32
        )
        model.get_joint_angles(joint_angles)

        robot_renderer.get_robot_mesh(joint_angles + 1) # warmup
        start_time = time.time()
        robot_mesh = robot_renderer.get_robot_mesh(joint_angles)
        end_time = time.time()
        print(f"Mesh computing time: {(end_time - start_time) * 1000 :.4f} ms")

        # a.3) Generate all initial cTr in some way (N total). For demo, let's do random.
        N = args.sample_number
        cTr_inits = []
        for i in range(N):
            camera_roll_local = torch.empty(1).uniform_(
                0, 360
            )  # Random values in [0, 360]
            camera_roll = torch.empty(1).uniform_(0, 360)  # Random values in [0, 360]
            azimuth = torch.empty(1).uniform_(0, 360)  # Random values in [0, 360]
            elevation = torch.empty(1).uniform_(
                90 - 60, 90 - 30
            )  # Random values in [90-25, 90+25]
            # elevation = 30
            distance = torch.empty(1).uniform_(0.10, 0.17)

            pose_matrix = model.from_lookat_to_pose_matrix(
                distance, elevation, camera_roll_local
            )
            roll_rad = torch.deg2rad(camera_roll)  # Convert roll angle to radians
            roll_matrix = torch.tensor(
                [
                    [torch.cos(roll_rad), -torch.sin(roll_rad), 0],
                    [torch.sin(roll_rad), torch.cos(roll_rad), 0],
                    [0, 0, 1],
                ]
            )
            pose_matrix[:, :3, :3] = torch.matmul(roll_matrix, pose_matrix[:, :3, :3])
            cTr = model.pose_matrix_to_cTr(pose_matrix)
            if not torch.any(torch.isnan(cTr)):
                cTr_inits.append(cTr)
        cTr_batch = torch.cat(cTr_inits, dim=0) # All samples in a single batch
        B = cTr_batch.shape[0] 
        print(f"All ctr candiates: {cTr_batch.shape}")

        angle_axis_batch = cTr_batch[:, :3]  # Extract axis-angle part
        mix_angle_batch = axis_angle_to_mix_angle(angle_axis_batch)  # Convert to Euler angles
        axis_angle_converted = mix_angle_to_axis_angle(mix_angle_batch)

        # Test the conversion accuracy
        R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(angle_axis_batch)
        R_converted = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle_converted)

        # Check if the conversion is consistent
        rot_diff = torch.norm(R - R_converted, dim=(1, 2))
        print("Rotation matrix difference:", rot_diff)
        assert torch.allclose(R, R_converted, atol=1e-5)
        print("Axis-angle conversion is consistent.")

        # a.4) Render silhouette shaders
        # warm-up for more accurate timing
        pred_masks = model.render_robot_mask_batch(
            cTr_batch, robot_mesh, robot_renderer
        ) # shape is [B, H, W]

        start_time = time.time()
        pred_masks = model.render_robot_mask_batch(
            cTr_batch, robot_mesh, robot_renderer
        ) # shape is [B, H, W]
        end_time = time.time()

        print(f"Predicted masks: {pred_masks.shape}")
        print(f"Batch Rendering Time (PyTorch3D): {(end_time - start_time) * 1000 :.4f} ms")

        # b.1) Configure NvDiffRast renderer
        glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
        resolution = (args.height, args.width)

        # b.2) Prepare data for rendering (using utils in PyTorch3D)
        # mix_angle_batch[:,2] += torch.pi / 3 # local roll
        # axis_angle_converted = mix_angle_to_axis_angle(mix_angle_batch)  # Convert back to axis-angle
        cTr_batch[:, :3] = axis_angle_converted  # Update the axis-angle part with converted values

        R_batched = kornia.geometry.conversions.angle_axis_to_rotation_matrix(
            cTr_batch[:, :3]
        ) 
        R_batched = R_batched.transpose(1, 2)
        T_batched = cTr_batch[:, 3:] 
        negative_mask = T_batched[:, -1] < 0  #flip where negative_mask is True
        T_batched_ = T_batched.clone()
        T_batched_[negative_mask] = -T_batched_[negative_mask]
        R_batched_ = R_batched.clone()
        R_batched_[negative_mask] = -R_batched_[negative_mask]
        pos, pos_idx = transform_mesh(
            cameras=robot_renderer.cameras, mesh=robot_mesh.extend(B),
            R=R_batched_, T=T_batched_, args=args
        ) # project the batched meshes in the clip space
        
        # Check if all instance in pos_idx are the same
        for i in range(1, len(pos_idx)):
            assert torch.all(pos_idx[0] == pos_idx[i]), "Different instance indices in the batch"
        
        # b.3) Render the silhouette images
        # warm-up for more accurate timing
        pred_masks_nv = render(glctx, pos, pos_idx[0], resolution) # instance mode, all topologies (pos_idx) are the same
        start_time = time.time()
        pred_masks_nv = render(glctx, pos, pos_idx[0], resolution) # shape is [B, H, W]
        end_time = time.time()

        # Project the origin (the translation vector) to the image plane
        fx, fy = args.fx, args.fy
        px, py = args.px, args.py

        # View-space coordinates of the origin in camera frame (R @ 0 + T)
        origin_camera = T_batched_  # (B, 3)

        # Project to pixel coordinates
        x = fx * (origin_camera[:, 0] / origin_camera[:, 2]) + px
        y = fy * (origin_camera[:, 1] / origin_camera[:, 2]) + py

        origin_proj = torch.stack([x, y], dim=-1)  # shape [B, 2]
        origin_proj_int = origin_proj.round().to(torch.int32)

        print(f"Predicted masks (NvDiffRast): {pred_masks_nv.shape}")
        print(f"Batch Rendering Time (NvDiffRast): {(end_time - start_time) * 1000 :.4f} ms")
        
        # Display the images 
        pred_masks = pred_masks.cpu().numpy()
        pred_masks_nv = pred_masks_nv.cpu().numpy()
        for i in range(min(0, args.sample_number)):
            img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            img[..., 0] = (pred_masks[i] * 255).astype(np.uint8) # blue for PyTorch3D
            img[..., 2] = (pred_masks_nv[i] * 255).astype(np.uint8) # red for NvDiffRast
            # Draw the origin point
            u, v = origin_proj_int[i]
            if 0 <= u < args.width and 0 <= v < args.height:
                cv2.circle(img, (u.item(), v.item()), 5, (0, 255, 0), -1)  # green for origin
            cv2.putText(img, f"Sample {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Predicted Masks", img)
            cv2.waitKey(0)

        # Test unscented mix angle to axis-angle conversion
        for i in range(args.sample_number):
            test_mix_angle = mix_angle_batch[i]
            stdev = torch.tensor([1e-2, 1e-1, 1e-2], dtype=torch.float32).cuda()
            mean_axis_angle, cov_axis_angle = unscented_mix_angle_to_axis_angle(test_mix_angle, stdev)
            print("Mean:\n", mean_axis_angle)
            print("Covariance:\n", cov_axis_angle)

            # Compare with empirical mean and covariance
            generated_mix_angles = test_mix_angle + stdev * torch.randn(1000, 3).cuda()
            generated_axis_angles = mix_angle_to_axis_angle(generated_mix_angles)
            empirical_mean = generated_axis_angles.mean(dim=0)
            empirical_cov = torch.cov(generated_axis_angles.T)
            print("Empirical Mean:\n", empirical_mean)
            print("Empirical Covariance:\n", empirical_cov)

            # Compute a diagonal stdev based on d_ii = sqrt(a_ii + \sum_{i\not=j} a_ij)
            diag = torch.diag(cov_axis_angle)  # shape (n,)
            off_diag_sum = torch.sum(torch.abs(cov_axis_angle), dim=1) - torch.abs(diag)
            diagonal_stdev = torch.sqrt(diag + off_diag_sum)
            print("Diagonal Stdev:\n", diagonal_stdev)

            # # Use CVXPY to find the smallest diagonal matrix dominating the covariance
            # A = cov_axis_angle.cpu().numpy()
            # D_opt = smallest_diag_dominating_A_cvxpy(A)
            # opt_diagonal_stdev = torch.tensor(np.sqrt(np.diag(D_opt)), dtype=torch.float32).cuda()
            # print("Optimal Diagonal Stdev (CVXPY):\n", opt_diagonal_stdev)