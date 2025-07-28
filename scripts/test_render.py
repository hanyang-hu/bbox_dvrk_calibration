import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
    unscented_mix_angle_to_axis_angle,
    enforce_quaternion_consistency,
    enforce_axis_angle_consistency
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


def main1():
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
            # print("Mean:\n", mean_axis_angle)
            # print("Covariance:\n", cov_axis_angle)

            # Compare with empirical mean and covariance
            generated_mix_angles = test_mix_angle + stdev * torch.randn(1000, 3).cuda()
            generated_axis_angles = mix_angle_to_axis_angle(generated_mix_angles)
            empirical_mean = generated_axis_angles.mean(dim=0)
            empirical_cov = torch.cov(generated_axis_angles.T)
            # print("Empirical Mean:\n", empirical_mean)
            # print("Empirical Covariance:\n", empirical_cov)

            # Compute a diagonal stdev based on d_ii = sqrt(a_ii + \sum_{i\not=j} a_ij)
            diag = torch.diag(cov_axis_angle)  # shape (n,)
            off_diag_sum = torch.sum(torch.abs(cov_axis_angle), dim=1) - torch.abs(diag)
            diagonal_stdev = torch.sqrt(diag + off_diag_sum)
            # print("Diagonal Stdev:\n", diagonal_stdev)

            # Conduct SVD to the cov_axis_angle, output U and Sigma      
            L, Q = torch.linalg.eigh(empirical_cov)
            # print("Empirical Covariance:\n", empirical_cov)
            if (L**0.5 > 0.1).any():
                print("Eigenvalue^0.5:\n", L**0.5)
            # print("Eigenvector:\n", Q)

            # # Convert mix angle to axis angle, then to quaternions, compute empirical mean and covariance
            # empirical_quat = kornia.geometry.conversions.axis_angle_to_quaternion(generated_axis_angles)
            # empirical_quat = enforce_quaternion_consistency(empirical_quat)  # Ensure sign consistency
            # empirical_mean_quat = empirical_quat.mean(dim=0)
            # empirical_cov_quat = torch.cov(empirical_quat.T)
            # # print("Empirical Mean Quaternion:\n", empirical_mean_quat)
            # # print("Empirical Covariance Quaternion:\n", empirical_cov_quat)
            # diag = torch.diag(empirical_cov_quat)
            # off_diag_sum = torch.sum(torch.abs(empirical_cov_quat), dim=1) - torch.abs(diag)
            # diagonal_stdev_quat = torch.sqrt(diag + off_diag_sum)
            # # print("Diagonal Stdev Quaternion:\n", diagonal_stdev_quat)
            # if (diagonal_stdev_quat > 0.01).any():
            #     print(f"Warning: High diagonal stdev for sample {i+1}: {diagonal_stdev_quat}")
            #     print()


def main2():
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
        N = 2
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
        
        cTr_start = torch.cat([cTr_batch[0,:3], cTr_batch[1,3:]], dim=0)  # (1, 6)
        cTr_end = cTr_batch[1] # (1, 6)
        cTr_start = cTr_start * 0.2 + cTr_end * 0.8  # Mix the two poses
        # print("Starting Pose:", cTr_start)
        # print("Ending Pose:", cTr_end)

        # Convert to axis-angle, mix angle, and quaternion
        axis_angle_start = cTr_start[:3].unsqueeze(0)  # (1, 3)
        axis_angle_end = cTr_end[:3].unsqueeze(0)
        mix_angle_start = axis_angle_to_mix_angle(axis_angle_start)
        mix_angle_end = axis_angle_to_mix_angle(axis_angle_end)
        quaternion_start = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle_start)
        quaternion_end = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle_end)
        if (quaternion_start[0,1:] * quaternion_end[0,1:]).sum(dim=-1).item() < 0:
            quaternion_end = -quaternion_end

        print("Axis-Angle Start:", axis_angle_start)
        print("Axis-Angle End:", axis_angle_end)
        print("Mix Angle Start:", mix_angle_start)
        print("Mix Angle End:", mix_angle_end)
        print("Quaternion Start:", quaternion_start)
        print("Quaternion End:", quaternion_end)

        # Interpolate and convert back to axis angles
        num_steps = 30
        axis_angle_interpolated = torch.lerp(axis_angle_start, axis_angle_end, torch.linspace(0, 1, num_steps).unsqueeze(1).cuda())
        mix_angle_interpolated = torch.lerp(mix_angle_start, mix_angle_end, torch.linspace(0, 1, num_steps).unsqueeze(1).cuda())
        # quaternion_interpolated = torch.slerp(quaternion_start, quaternion_end, torch.linspace(0, 1, num_steps).unsqueeze(1).cuda())
        q1 = kornia.geometry.quaternion.Quaternion(quaternion_start)
        q2 = kornia.geometry.quaternion.Quaternion(quaternion_end)
        quaternion_interpolated = []
        for t in torch.linspace(0, 1, num_steps):
            q3 = q1.slerp(q2, t)
            quaternion_interpolated.append(q3.q)
        quaternion_interpolated = torch.stack(quaternion_interpolated, dim=0).squeeze()

        axis_angle_interpolated_from_mix = mix_angle_to_axis_angle(mix_angle_interpolated)
        axis_angle_interpolated_from_quat = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion_interpolated)

        cTr_axis_angle = torch.cat([axis_angle_interpolated, cTr_batch[0,3:].unsqueeze(0).repeat(num_steps, 1)], dim=-1)
        cTr_mix_angle = torch.cat([axis_angle_interpolated_from_mix, cTr_batch[0,3:].unsqueeze(0).repeat(num_steps, 1)], dim=-1)
        cTr_quaternion = torch.cat([axis_angle_interpolated_from_quat, cTr_batch[0,3:].unsqueeze(0).repeat(num_steps, 1)], dim=-1)

        # Render silhouette shaders of the interpolated poses
        # b.1) Configure NvDiffRast renderer
        glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
        resolution = (args.height, args.width)

        # b.2) Render for axis angle interpolation

        def get_pos_and_idx(cTr_batch, num_steps=num_steps):
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
                cameras=robot_renderer.cameras, mesh=robot_mesh.extend(num_steps),
                R=R_batched_, T=T_batched_, args=args
            ) # project the batched meshes in the clip space
            
            # Check if all instance in pos_idx are the same
            for i in range(1, len(pos_idx)):
                assert torch.all(pos_idx[0] == pos_idx[i]), "Different instance indices in the batch"
            
            return pos, pos_idx

        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        fps = 10  # Adjust as needed

        pos, pos_idx = get_pos_and_idx(cTr_axis_angle)

        # output_path = "./interpolation_visualization/axis_angle_interpolation.mp4"
        # video_writer_axis = cv2.VideoWriter(output_path, fourcc, fps, (args.width, args.height))
        
        pred_masks_axis = render(glctx, pos, pos_idx[0], resolution) # instance mode, all topologies (pos_idx) are the same

        # pred_masks = pred_masks_axis.cpu().numpy()
        # for i in range(args.sample_number):
        #     img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        #     img[..., 0] = (pred_masks[i] * 255).astype(np.uint8) # blue for current frame
        #     # if i > 0:
        #     #     img[..., 2] = (pred_masks[i-1] * 255).astype(np.uint8) # red for previous frame
        #     cv2.putText(img, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #     # cv2.imshow("Rendered Masks for Axis Angle Interpolation", img)
        #     # cv2.waitKey(0)
        #     video_writer_axis.write(img)

        # b.3) Render for mix angle interpolation
        pos, pos_idx = get_pos_and_idx(cTr_mix_angle)

        # output_path = "./interpolation_visualization/transformed_angle_interpolation.mp4"
        # video_writer_mix = cv2.VideoWriter(output_path, fourcc, fps, (args.width, args.height))
        
        pred_masks_mix = render(glctx, pos, pos_idx[0], resolution) # instance mode, all topologies (pos_idx) are the same

        # pred_masks = pred_masks_mix.cpu().numpy()
        # for i in range(args.sample_number):
        #     img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        #     img[..., 0] = (pred_masks[i] * 255).astype(np.uint8) # blue for current frame
        #     # if i > 0:
        #     #     img[..., 2] = (pred_masks[i-1] * 255).astype(np.uint8) # red for previous frame
        #     cv2.putText(img, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #     # cv2.imshow("Rendered Masks for Mix Angle Interpolation", img)
        #     # cv2.waitKey(0)
        #     video_writer_mix.write(img)

        # b.4) Render for quaternion interpolation
        pos, pos_idx = get_pos_and_idx(cTr_quaternion)

        # output_path = "./interpolation_visualization/quaternion_interpolation.mp4"
        # video_writer_quats = cv2.VideoWriter(output_path, fourcc, fps, (args.width, args.height))
        
        pred_masks_quat = render(glctx, pos, pos_idx[0], resolution) # instance mode, all topologies (pos_idx) are the same

        # pred_masks = pred_masks_quat.cpu().numpy()
        # for i in range(args.sample_number):
        #     img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        #     img[..., 0] = (pred_masks[i] * 255).astype(np.uint8) # blue for current frame
        #     # if i > 0:
        #     #     img[..., 2] = (pred_masks[i-1] * 255).astype(np.uint8) # red for previous frame
        #     cv2.putText(img, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #     # cv2.imshow("Rendered Masks for Quaternion Interpolation", img)
        #     # cv2.waitKey(0)
        #     video_writer_quats.write(img)

        # Render all interpolated poses in a single video
        output_path = "./interpolation_visualization/all_interpolations.mp4"
        video_writer_all = cv2.VideoWriter(output_path, fourcc, fps, (args.width, args.height))

        pred_masks_axis = pred_masks_axis.cpu().numpy()
        pred_masks_mix = pred_masks_mix.cpu().numpy()
        pred_masks_quat = pred_masks_quat.cpu().numpy()

        for i in range(args.sample_number):
            img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            img[..., 0] = (pred_masks_axis[i] * 255).astype(np.uint8) 
            img[..., 1] = (pred_masks_mix[i] * 255).astype(np.uint8) + (pred_masks_axis[i] * 200).astype(np.uint8) // 2  # mix color
            img[..., 2] = (pred_masks_quat[i] * 255).astype(np.uint8) + (pred_masks_axis[i] * 100).astype(np.uint8) // 2  # mix color

            cv2.putText(img, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # puttext to label the colors (light blue for axis angle, green for mix angle, red for quaternion)
            cv2.putText(img, "Axis Angle", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            cv2.putText(img, "Transformed Angle", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, "Quaternion", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.imshow("All Interpolations", img)
            # cv2.waitKey(0)
            video_writer_all.write(img)


        # video_writer_axis.release()
        # video_writer_mix.release()
        # video_writer_quats.release()
        video_writer_all.release()
        print(f"Saved video to ./interpolation_visualization/")




if __name__ == "__main__":
    main1()
    # main2()