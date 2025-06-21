import argparse
import numpy as np
import torch 
import os
import glob
import cv2
import time
import sys
import os
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.utils.detection_utils import detect_lines
from diffcali.eval_dvrk.batch_optimize import BatchOptimize  # The class we just wrote
from diffcali.eval_dvrk.optimize import Optimize  # Your single-sample class
from diffcali.eval_dvrk.black_box_optimize import BlackBoxOptimize

from diffcali.eval_dvrk.trackers import GradientTracker, EvoTracker

from evotorch.tools.misc import RealOrVector # Union[float, Iterable[float], torch.Tensor]

trackers = {
    "gradient": GradientTracker,
    "evolution": EvoTracker,
}

torch.cuda.empty_cache()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument("--data_dir", type=str, default="data/consecutive_prediction/")  
    parser.add_argument("--difficulty", type=str, default="0617")
    parser.add_argument("--frame_start", type=int, default=140) 
    parser.add_argument("--frame_end", type=int, default=160)  # End frame for the sequence
    parser.add_argument("--batch_opt_lr", type=float, default=3e-3)
    parser.add_argument("--single_opt_lr", type=float, default=5e-4) # if using gradient descent
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--batch_iters", type=int, default=100
    )  # Coarse steps per batch
    parser.add_argument(
        "--final_iters", type=int, default=1000
    )  # Final single-sample refine
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=200)
    parser.add_argument("--use_nvdiffrast", action="store_true")
    parser.add_argument("--use_bbox_optimizer", action="store_true") # Use XNES for initialization
    parser.add_argument("--tracker", type=str, default="evolution", choices=list(trackers.keys()))
    parser.add_argument("--online_iters", type=int, default=10)  # Number of iterations for online tracking
    parser.add_argument("--tracking_visualization", action="store_true")  # Whether to visualize the tracking process
    parser.add_argument("--online_lr", type=float, default=1e-3)  # Learning rate for online tracking

    stdev_init = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32).cuda() # Initial standard deviation for XNES/SNES
    stdev_init[:3] *= 1e-1 # angles (3D)
    stdev_init[3:6] *= 1e-3 # translations (3D)
    stdev_init[6:] *= 1e-3 # joint angles (4D)
    parser.add_argument("--stdev_init", type=RealOrVector, default=stdev_init)  # Standard deviation for initial noise in XNES

    parser.add_argument("--log_interval", type=int, default=1000)  # Logging interval for optimization
    args = parser.parse_args()

    return args


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707
    args.scale = 1.0

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


def buildcTr(cTr_train, cTr_nontrain):
    # Rebuild [angle_axis(3), xyz(3)] from cTr_train
    return th.cat([cTr_train[0], cTr_train[1], cTr_train[2]], dim=0)


def display_data(data_lst, idx):
    data, idx = data_lst[idx], idx % len(data_lst)
    print(f"Frame {idx}:")
    print(f"  Frame path: {data['ref_mask_path']}")
    print(f"  Ref keypoints: {data['ref_keypoints']}")
    print(f"  Joint angles: {data['joint_angles']}")
    print(f"  Optimized cTr: {data['optim_ctr']}")
    print(f"  Optimized joint angles: {data['optim_joint_angles']}")


def read_data(args):
    """
    Read the frames and relevant data from the data directory.
    """
    data_lst = []

    data_dir = os.path.join(args.data_dir, args.difficulty)
    for i in range(args.frame_start, args.frame_end):
        frame_dir = os.path.join(data_dir, f"{i}")

        # Find the mask
        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        if len(mask_lst) == 0:
            raise ValueError(f"No mask found in {frame_dir}")
        if len(mask_lst) > 1:
            raise ValueError(f"Multiple masks found in {frame_dir}")

        mask_path = mask_lst[0]
        # frame = cv2.imread(frame_path)
        XXXX = mask_path.split("/")[-1].split(".")[0][1:]

        # Read ref_img_file of name 0XXXX.jpg
        ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")
        ref_img = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise ValueError(f"No ref_img found in {frame_dir}")
        ref_mask = (ref_img / 255.0).astype(np.float32)
        ref_mask = th.tensor(ref_mask, requires_grad=False, dtype=th.float32).cuda()

        # Get reference key points
        ref_keypoints = get_reference_keypoints_auto(ref_mask_path, num_keypoints=2)
        ref_keypoints = torch.tensor(ref_keypoints).squeeze().float().cuda()

        # Get joint angles
        joint_path = os.path.join(frame_dir, "joint_" + XXXX + ".npy")
        jaw_path = os.path.join(frame_dir, "jaw_" + XXXX + ".npy")
        if not os.path.exists(joint_path):
            raise ValueError(f"No joint angles found in {frame_dir}")
        if not os.path.exists(jaw_path):
            raise ValueError(f"No jaw angles found in {frame_dir}")
        joints = np.load(joint_path)
        jaw = np.load(jaw_path)
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = th.tensor(
            joint_angles_np, requires_grad=False, dtype=th.float32
        ).cuda() 

        # Get optimized pose and joint angles
        optim_ctr_path = os.path.join(frame_dir, "optimized_ctr.npy")
        optim_joint_path = os.path.join(frame_dir, "optimized_joint_angles.npy")
        optim_ctr_np = np.load(optim_ctr_path)
        optim_joint_angles_np = np.load(optim_joint_path)
        optim_ctr = th.tensor(
            optim_ctr_np, requires_grad=False, dtype=th.float32
        ).cuda()
        optim_joint_angles = th.tensor(
            optim_joint_angles_np, requires_grad=False, dtype=th.float32
        ).cuda()

        data = {
            # "frame": frame,
            "ref_mask": ref_mask.clone(),
            "ref_mask_path": ref_mask_path,
            "ref_keypoints": ref_keypoints.clone(),
            "joint_angles": joint_angles.clone(),
            "optim_ctr": optim_ctr.clone(),
            "optim_joint_angles": optim_joint_angles.clone(),
        }
        # print(data["joint_angles"], joints, jaw)
        data_lst.append(data)

    mesh_files = [
        f"{args.mesh_dir}/shaft_multi_cylinder.ply",
        f"{args.mesh_dir}/logo_low_res_1.ply",
        f"{args.mesh_dir}/jawright_lowres.ply",
        f"{args.mesh_dir}/jawleft_lowres.ply",
    ]

    return data_lst, mesh_files


def initialization(model, init_data, mesh_files):
    """
    Use the method in origin_retracing.py to initialize the pose and joint angles.
    """
    joint_angles = init_data["joint_angles"]
    joint_angles_read = joint_angles.clone()
    model.get_joint_angles(joint_angles)
    robot_mesh = robot_renderer.get_robot_mesh(joint_angles)
    ref_keypoints = init_data["ref_keypoints"]

    # Get the initial cTr
    N = args.sample_number
    cTr_inits = []
    for i in range(N):
        camera_roll_local = th.empty(1).uniform_(
            0, 360
        )  # Random values in [0, 360]
        camera_roll = th.empty(1).uniform_(0, 360)  # Random values in [0, 360]
        azimuth = th.empty(1).uniform_(0, 360)  # Random values in [0, 360]
        elevation = th.empty(1).uniform_(
            90 - 60, 90 - 30
        )  # Random values in [90-25, 90+25]
        # elevation = 30
        distance = th.empty(1).uniform_(0.10, 0.17)

        pose_matrix = model.from_lookat_to_pose_matrix(
            distance, elevation, camera_roll_local
        )
        roll_rad = th.deg2rad(camera_roll)  # Convert roll angle to radians
        roll_matrix = th.tensor(
            [
                [th.cos(roll_rad), -th.sin(roll_rad), 0],
                [th.sin(roll_rad), th.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )
        pose_matrix[:, :3, :3] = th.matmul(roll_matrix, pose_matrix[:, :3, :3])
        cTr = model.pose_matrix_to_cTr(pose_matrix)
        if not th.any(th.isnan(cTr)):
            cTr_inits.append(cTr)
    cTr_inits_t = th.cat(cTr_inits, dim=0)
    print(f"All ctr candiates: {cTr_inits_t.shape }")
    bsz = args.batch_size
    final_batch_winners = []
    final_batch_winners_losses = []

    # Batch optimization
    if N <= bsz:
        # Handle small N case (process in a single batch)
        print(f"Small N={N}, processing in a single batch.")
        cTr_batch = cTr_inits_t  # All samples in a single batch
        B = cTr_batch.shape[0]  # Batch size is equal to N

        batch_opt = BatchOptimize(
            cTr_batch=cTr_batch,
            joint_angles=joint_angles,
            model=model,
            robot_mesh=robot_mesh,
            robot_renderer=robot_renderer,
            ref_keypoints=ref_keypoints,
            fx=ctrnet_args.fx,
            fy=ctrnet_args.fy,
            px=ctrnet_args.px,
            py=ctrnet_args.py,
            lr=args.batch_opt_lr,
            batch_size=B,
        )

        batch_opt.readRefImage(init_data["ref_mask_path"])

        # Optimize
        best_cTr_in_batch, best_loss_in_batch = batch_opt.optimize_batch(
            iterations=args.batch_iters, grid_search=False, ld1=3, ld2=3, ld3=3
        )

        final_batch_winners.append(best_cTr_in_batch)
        final_batch_winners_losses.append(best_loss_in_batch)
        print(
            f"[Single batch] best loss={best_loss_in_batch:.4f}, ctr={best_cTr_in_batch}"
        )
    else:

        for start in range(0, N, bsz):
            end = min(start + bsz, N)
            cTr_batch = cTr_inits_t[start:end]  # shape (B,6)
            B = cTr_batch.shape[0]

            batch_opt = BatchOptimize(
                cTr_batch=cTr_batch,
                joint_angles=joint_angles,
                model=model,
                robot_mesh=robot_mesh,
                robot_renderer=robot_renderer,
                ref_keypoints=ref_keypoints,
                fx=ctrnet_args.fx,
                fy=ctrnet_args.fy,
                px=ctrnet_args.px,
                py=ctrnet_args.py,
                lr=args.batch_opt_lr,
                batch_size=bsz,
            )

            batch_opt.readRefImage(init_data["ref_mask_path"])
            # Coarse optimize
            best_cTr_in_batch, best_loss_in_batch = batch_opt.optimize_batch(
                iterations=args.batch_iters, grid_search=False, ld1=3, ld2=3, ld3=3
            )

            # final_cTr shape => (B,6), final_losses => (B,), final_angles => (B,)
            # Pick best from this batch

            final_batch_winners.append(best_cTr_in_batch)
            final_batch_winners_losses.append(best_loss_in_batch)
            print(
                f"[Batch range {start}-{end}] best in batch => loss={best_loss_in_batch:.4f} ctr={best_cTr_in_batch}"
            )

    # 6) Global best
    final_batch_winners_losses_np = np.array(final_batch_winners_losses)
    best_idx_global = np.argmin(final_batch_winners_losses_np)
    best_global_loss = final_batch_winners_losses_np[best_idx_global]
    best_global_cTr = final_batch_winners[best_idx_global]
    print("==== Global best among all batches ====")
    print("loss=", best_global_loss, "cTr=", best_global_cTr.cpu().numpy())

    # Additional: add gaussian noise into the best ctr and rank in the new batch.....
    noisy_bsz = args.batch_size
    temp = best_global_cTr.expand(noisy_bsz, best_global_cTr.shape[-1])  # (B, 6)
    noise = th.randn_like(temp)
    angle_std_scale = 0.1
    xyz_std_scale = 0.00001
    noise[:, :3] *= angle_std_scale  # Scale angles
    noise[:, 3:] *= xyz_std_scale  # Scale translations
    noisy_ctr = temp + noise  # (B, 6)

    nsy_opt = BatchOptimize(
        cTr_batch=noisy_ctr,
        joint_angles=joint_angles,
        model=model,
        robot_mesh=robot_mesh,
        robot_renderer=robot_renderer,
        ref_keypoints=ref_keypoints,
        fx=ctrnet_args.fx,
        fy=ctrnet_args.fy,
        px=ctrnet_args.px,
        py=ctrnet_args.py,
        lr=args.batch_opt_lr,
        batch_size=noisy_bsz,
    )

    nsy_opt.readRefImage(init_data["ref_mask_path"])
    # Coarse optimize
    best_cTr, best_loss = nsy_opt.optimize_batch(
        iterations=args.batch_iters, grid_search=False, ld1=3, ld2=3, ld3=3
    )
    print("==== Global best among noisy cTrs ====")
    print("loss=", best_loss, "cTr=", best_cTr)

    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    # Refinement
    if args.use_bbox_optimizer:
        with th.no_grad():
            bbox_opt = BlackBoxOptimize(
                model=model,
                robot_mesh=robot_mesh,
                robot_renderer=robot_renderer,
                ref_keypoints=ref_keypoints,
                ref_mask_file=init_data["ref_mask_path"],
                joint_angles=joint_angles,
                fx=ctrnet_args.fx,
                fy=ctrnet_args.fy,
                px=ctrnet_args.px,
                py=ctrnet_args.py,
                ld1=3,
                ld2=3,
                ld3=3,
                center_init=th.cat([best_cTr, joint_angles], dim=0),
                log_interval=args.log_interval,
            )

            final_cTr_s, final_loss_s, joint_angles = bbox_opt.optimize(args.final_iters)

    else:
        # best_cTr_np = best_global_cTr.squeeze().cpu().numpy()
        best_cTr_np = best_cTr.squeeze().cpu().numpy()
        axis_angle = th.tensor(best_cTr_np[:3], device=model.device, requires_grad=True)
        xy = th.tensor(best_cTr_np[3:5], device=model.device, requires_grad=True)
        z = th.tensor(best_cTr_np[5:], device=model.device, requires_grad=True)
        joint_angles.requires_grad_(True) # make joint angles learnable

        model.get_joint_angles(joint_angles)

        cTr_train = [axis_angle, xy, z, joint_angles]

        single_opt = Optimize(
            cTr_train=cTr_train,
            model=model,
            robot_mesh=robot_mesh,
            robot_renderer=robot_renderer,
            lr=args.single_opt_lr,
            cTr_nontrain=None,
            buildcTr=buildcTr,
        )

        single_opt.readRefImage(init_data["ref_mask_path"])
        single_opt.ref_keypoints = ref_keypoints
        single_opt.ref_keypoints = th.tensor(
            single_opt.ref_keypoints, device=single_opt.model.device, dtype=th.float32
        )
        single_opt.fx, single_opt.fy, single_opt.px, single_opt.py = (
            ctrnet_args.fx,
            ctrnet_args.fy,
            ctrnet_args.px,
            ctrnet_args.py,
        )
        saving_dir = os.path.join(args.data_dir, "optimization")
        os.makedirs(saving_dir, exist_ok=True)

        final_cTr_s, final_loss_s, final_angle_s = single_opt.optimize(
            iterations=args.final_iters,
            save_fig_dir=saving_dir,
            ld1=3,
            ld2=3,
            ld3=3,
            set2=[3, 3, 3],
            xyz_steps=1,
            angles_steps=3,
            saving_interval=args.log_interval,
            coarse_step_num=300,
            grid_search=False,
        )

    print("==== Initialization results ====")
    print(f"  Refined cTr = {final_cTr_s}")
    print(f"  Refined loss = {final_loss_s}")
    print(f"  joint angles before: {joint_angles_read}")
    print(f"  joint angles after: {joint_angles}")

    return final_cTr_s, joint_angles


if __name__ == "__main__":
    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    ctrnet_args.use_nvdiffrast = args.use_nvdiffrast
    if ctrnet_args.use_nvdiffrast:
        print("Using NvDiffRast!")

    # Obtain the data
    data_lst, mesh_files = read_data(args)

    # Display the data (except for the images) for the first and last frames
    display_data(data_lst, 0)
    display_data(data_lst, -1)

    # Build the model
    model = CtRNet(ctrnet_args)
    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    # Initialize the model
    cTr, joint_angles = initialization(
        model, data_lst[0], mesh_files
    )  

    # Camera intrinsic matrix
    intr = torch.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px], 
            [0, ctrnet_args.fy, ctrnet_args.py], 
            [0, 0, 1]
        ],
        device="cuda",
        dtype=joint_angles.dtype,
    )

    p_local1 = (
        torch.tensor([0.0, 0.0004, 0.009])
        .to(joint_angles.dtype)
        .to(model.device)
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, 0.009])
        .to(joint_angles.dtype)
        .to(model.device)
    )
  
    print(f"==== Tracking results ====")

    gc.collect()
    torch.cuda.empty_cache()

    tracker = trackers[args.tracker](
        model=model,
        robot_renderer=robot_renderer,
        init_cTr=cTr,
        init_joint_angles=joint_angles,
        num_iters=args.online_iters,
        intr=intr,
        p_local1=p_local1,
        p_local2=p_local2,
    )

    if args.tracker == "gradient":
        tracker.lr = args.online_lr  # Set the learning rate for gradient descent
    if args.tracker == "evolution":
        tracker.stdev_init = args.stdev_init

    # Track the rest of the frames
    loss_lst, time_lst = [], []
    for i in range(1, len(data_lst)):
        # print(data_lst[i]["joint_angles"])

        start_time = time.time()
        cTr, joint_angles, loss, overlay = tracker.track(
            mask=data_lst[i]["ref_mask"],
            joint_angles=data_lst[i]["joint_angles"],
            ref_keypoints=data_lst[i]["ref_keypoints"],
            visualization=args.tracking_visualization,
        )
        end_time = time.time()

        # Save the overlay image
        if args.tracking_visualization:
            overlay_path = os.path.join("./tracking/", f"overlay_{i}.png")
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            cv2.imwrite(overlay_path, overlay)

        loss_lst.append(loss)
        time_lst.append(end_time - start_time)

        print(f"Frame {i} - Loss: {loss:.4f}, Time: {end_time - start_time:.4f} seconds")

    # Print the average MSE and time
    avg_loss = np.mean(loss_lst)
    avg_time = np.mean(time_lst)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Time: {avg_time:.4f} seconds")
    print("Tracking completed.")