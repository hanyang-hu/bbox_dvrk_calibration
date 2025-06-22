import torch
import torch as th
import torch.nn.functional as F
import numpy as np
import math
import cv2
import os

from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK
from diffcali.utils.projection_utils import *
from diffcali.utils.ui_utils import *
from diffcali.utils.cylinder_projection_utils import (
    projectCylinderTorch,
    transform_points,
    transform_points_b,
)
from diffcali.utils.detection_utils import detect_lines

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import SNES, XNES, CMAES
from evotorch.logging import Logger


torch.set_default_dtype(torch.float32)


# Loss settings
MSE_WEIGHT = 10.0  # weight for the MSE loss
PTS_WEIGHT = 1e-2  # weight for the keypoint loss
CYD_WEIGHT = 1e-2  # weight for the cylinder loss

USE_PTS_LOSS = True  # whether to use keypoint loss
USE_CYD_LOSS = False  # whether to use cylinder loss


def keypoint_loss_batch(keypoints_a, keypoints_b):
    """
    Computes the Chamfer distance between two sets of keypoints.

    Args:
        keypoints_a (torch.Tensor): Tensor of keypoints (shape: [B, 2, 2]).
        keypoints_b (torch.Tensor): Tensor of keypoints (shape: [B, 2, 2]).

    Returns:
        torch.Tensor: The computed Chamfer distance.
    """
    if keypoints_a.size(1) != 2 or keypoints_b.size(1) != 2:
        raise ValueError("This function assumes two keypoints per set in each batch.")

    # Permutation 1: A0->B0 and A1->B1
    dist_1 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 0], dim=1) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 1], dim=1
    )

    # Permutation 2: A0->B1 and A1->B0
    dist_2 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 1], dim=1) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 0], dim=1
    )

    # Choose the pairing that results in minimal distance for each batch
    min_dist = torch.min(dist_1, dist_2)  # [B]

    # Align the centerline for each batch
    centerline_loss = torch.norm(
        torch.mean(keypoints_a, dim=1) - torch.mean(keypoints_b, dim=1), dim=1
    )  # [B]

    return min_dist + centerline_loss


def cylinder_loss_params(detected_lines, projected_lines):
    """Input:
    detected_lines [B, 2, 2]
    projected_lines [B, 2, 2]

    Output:
    loss [B]
    """

    def to_theta_rho(lines):
        # lines: [B, 2, 2], so:
        #   lines[:, 0, :] -> line0 for each batch -> shape [B, 2]
        #   lines[:, 1, :] -> line1 for each batch -> shape [B, 2]
        a = lines[..., 0]  # shape [B, 2]
        b = lines[..., 1]  # shape [B, 2]
        n = (a**2 + b**2).sqrt() + 1e-9  # shape [B, 2]
        rho = 1.0 / n  # shape [B, 2]
        theta = torch.atan2(b, a)  # shape [B, 2], range [-pi, +pi]
        return theta, rho

    # 2) Per-line difference:  (theta1, rho1) vs. (theta2, rho2)
    #    Each is shape [B], returning shape [B].
    def line_difference(theta1, rho1, theta2, rho2):
        # Each input is [B], because we’ll call this once per line pairing.
        delta_theta = torch.abs(theta1 - theta2)  # [B]
        delta_theta = torch.min(
            delta_theta, 2 * math.pi - delta_theta
        )  # handle 2π periodicity
        delta_theta = torch.min(
            delta_theta, math.pi - delta_theta
        )  # optional, if you want symmetrical ranges
        delta_rho = torch.abs(rho1 - rho2)  # [B]
        # Return elementwise line distance
        # (Originally you did a mean, but now we keep it per-batch-sample.)
        dist = delta_rho + 0.7 * delta_theta  # [B]
        return dist, delta_theta

    # 3) Extract batched theta, rho for detected and projected lines
    theta_det, rho_det = to_theta_rho(detected_lines)  # each => shape [B, 2]
    theta_proj, rho_proj = to_theta_rho(projected_lines)  # each => shape [B, 2]

    # 4) Pairing 1: Det[0] ↔ Proj[0], Det[1] ↔ Proj[1]
    #    We'll index line 0: (theta_det[:, 0], rho_det[:, 0]) vs. (theta_proj[:, 0], rho_proj[:, 0])
    loss_1_0, theta_1_0 = line_difference(
        theta_det[:, 0], rho_det[:, 0], theta_proj[:, 0], rho_proj[:, 0]
    )  # each => [B]
    loss_1_1, theta_1_1 = line_difference(
        theta_det[:, 1], rho_det[:, 1], theta_proj[:, 1], rho_proj[:, 1]
    )  # each => [B]
    total_loss_1 = loss_1_0 + loss_1_1  # shape [B]

    # 5) Pairing 2: Det[0] ↔ Proj[1], Det[1] ↔ Proj[0]
    loss_2_0, theta_2_0 = line_difference(
        theta_det[:, 0], rho_det[:, 0], theta_proj[:, 1], rho_proj[:, 1]
    )
    loss_2_1, theta_2_1 = line_difference(
        theta_det[:, 1], rho_det[:, 1], theta_proj[:, 0], rho_proj[:, 0]
    )
    total_loss_2 = loss_2_0 + loss_2_1  # shape [B]

    # 6) Centerline alignment
    #    We take the mean over the lines dimension = 1, so each batch item has a single (theta, rho) average.
    theta_det_mean = torch.mean(theta_det, dim=1)  # shape [B]
    rho_det_mean = torch.mean(rho_det, dim=1)  # shape [B]
    theta_proj_mean = torch.mean(theta_proj, dim=1)
    rho_proj_mean = torch.mean(rho_proj, dim=1)

    centerline_loss, _ = line_difference(
        theta_det_mean, rho_det_mean, theta_proj_mean, rho_proj_mean
    )  # shape [B]

    # 7) Choose minimal pairing for each batch element, then add centerline loss
    #    total_loss_1, total_loss_2, and centerline_loss are all shape [B].
    line_loss = torch.where(total_loss_1 < total_loss_2, total_loss_1, total_loss_2)
    line_loss = line_loss + centerline_loss

    # line_loss is now [B].
    return line_loss  # [B]


class Tracker:
    def __init__(
        self, model, robot_renderer, init_cTr, init_joint_angles, 
        num_iters=5, intr=None, p_local1=None, p_local2=None, 
    ):
        self.model = model
        self.robot_renderer = robot_renderer

        self._prev_cTr = init_cTr
        self._prev_joint_angles = init_joint_angles
        self.num_iters = num_iters

        self.num_iters = num_iters # number of iterations for optimization

        self.intr = intr  
        self.p_local1 = p_local1  
        self.p_local2 = p_local2 

        self.fx, self.fy, self.px, self.py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()

    def track(self, mask, joint_angles, ref_keypoints, num_iters=5):
        """
        Inputs:
        - mask: The reference mask (from the segmentation module).
        - joint_angles: The joint angles read (may be inaccurate for cable-driven robots).
        - ref_keypoints: Keypoints in the reference image for tracking.
        - num_iters: Number of iterations for optimization.
        Outputs:
        - cTr: The 6D pose of the robot.
        - joint_angles: The optimized joint angles.
        - mse: The mean squared error between the rendered mask and the reference mask.
        - overlay: An overlay image of the predicted mask on the reference mask for visualization.
        """
        raise NotImplementedError("Tracking method not implemented.")

    def overlay_mask(self, ref_mask, pred_mask):
        """
        Overlay the predicted mask on the reference mask for visualization.
        """
        # Convert masks to grayscale images
        ref_mask = ref_mask.cpu().numpy()
        pred_mask = pred_mask.cpu().numpy()
        ref_mask = (ref_mask * 255).astype(np.uint8)
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        # Create a color overlay
        overlay = np.zeros((ref_mask.shape[0], ref_mask.shape[1], 3), dtype=np.uint8)
        overlay[..., 0] = ref_mask
        overlay[..., 1] = pred_mask

        return overlay


class GradientTracker(Tracker):
    def __init__(self, model, robot_renderer, init_cTr, init_joint_angles, num_iters=5, intr=None, p_local1=None, p_local2=None, lr=5e-4):
        super().__init__(model, robot_renderer, init_cTr, init_joint_angles, num_iters, intr, p_local1, p_local2)

        self.lr = lr # learning rate for Adam optimization

        # self.model.args.use_nvdiffrast = False # use PyTorch3D renderer instead

        if self.model.args.use_nvdiffrast and not self.model.use_antialiasing:
            print("[Antialiasing is not enabled in the NvDiffRast renderer. This may lead to inaccurate gradients.]")
            print("[Enabling antialiasing for better gradients.]")
            self.model.use_antialiasing = True # use antialiasing for better gradients

        self.kpts_loss = USE_PTS_LOSS
        self.cylinder_loss = USE_CYD_LOSS

        self.mse_weight = MSE_WEIGHT  # weight for the MSE loss
        self.pts_weight = PTS_WEIGHT  # weight for the keypoint loss
        self.cylinder_weight = CYD_WEIGHT  # weight for the cylinder loss

    def keypoint_chamfer_loss(self, keypoints_a, keypoints_b, pts=True):
        # Expand dimensions for broadcasting
        pts_a = keypoints_a
        pts_b = keypoints_b

        if keypoints_a.size(0) != 2 or keypoints_b.size(0) != 2:
            raise ValueError(
                "This brute force method only works for exactly two keypoints in each set."
            )

        # Compute distances for both possible permutations:
        # Permutation 1: A0->B0 and A1->B1
        dist_1 = th.norm(keypoints_a[0] - keypoints_b[0]) + th.norm(
            keypoints_a[1] - keypoints_b[1]
        )

        # Permutation 2: A0->B1 and A1->B0
        dist_2 = th.norm(keypoints_a[0] - keypoints_b[1]) + th.norm(
            keypoints_a[1] - keypoints_b[0]
        )

        # Choose the pairing that results in minimal distance
        min_dist = th.min(dist_1, dist_2)

        # Align the centerline:
        centerline_loss = th.norm(th.mean(pts_a, dim=0) - th.mean(pts_b, dim=0))

        # print(f"debugging loss scale....{chamfer_loss, parallelism_loss, distance_constraint_loss}")
        if pts == True:
            return min_dist + centerline_loss
        else:
            # print(f"checking the cylinder loss scale: {chamfer_loss, centerline_loss}")
            return min_dist

    def cylinder_loss_single(self, ref_mask, position, direction, pose_matrix, radius):
        # get the projected cylinder lines parameters

        intr = self.intr

        _, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
        cam_pts_3d_norm = th.nn.functional.normalize(cam_pts_3d_norm)
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, self.fx, self.fy, self.px, self.py
        )  # [1,2], [1,2]
        projected_lines = th.cat((e_1, e_2), dim=0)

        # print(f"debugging the line params: {detected_lines}")
        self.proj_line_params = projected_lines
        detected_lines = self.det_line_params

        # print(f"checking the shape of detected and projected lines {detected_lines}, {projected_lines}")  # [2, 2] [2, 2]
        cylinder_loss = cylinder_loss_params(detected_lines.unsqueeze(0), projected_lines.unsqueeze(0)) if detected_lines is not None else 0.0
        # parallelism = angle_between_lines(projected_lines, detected_lines)
        # print(f"testing line angle...{parallelism}")
        return cylinder_loss
        
    def track(self, mask, ref_img, joint_angles, visualization=False):
        ref_mask_np = mask.detach().cpu().numpy()

        if self.kpts_loss:
            ref_keypoints = get_reference_keypoints_auto(ref_img_path=None, ref_img=ref_img, num_keypoints=2)
            ref_keypoints = torch.tensor(ref_keypoints).squeeze().float().cuda()
        else:
            ref_keypoints = None

        if self.cylinder_loss:
            longest_lines = detect_lines(ref_mask_np, output=True) 

            longest_lines = np.array(longest_lines, dtype=np.float64)

            if longest_lines.shape[0] < 2:
                # Force skip cylinder or fallback
                print(
                    "WARNING: Not enough lines found by Hough transform. Skipping cylinder loss."
                )
                # You can set self.det_line_params to None or some fallback
                self.det_line_params = None

            else:
                # print(f"debugging the longest lines {longest_lines}")
                x1 = longest_lines[:, 0]
                y1 = longest_lines[:, 1]
                x2 = longest_lines[:, 2]
                y2 = longest_lines[:, 3]
                # print(f"debugging the end points x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                # Calculate line parameters (a, b, c) for detected lines
                a = y2 - y1
                b = x1 - x2
                c = x1 * y2 - x2 * y1  # Determinant for the line equation

                # Normalize to match the form au + bv = 1
                # norm = c + 1e-6  # Ensure no division by zero
                norm = np.abs(c)  # Compute the absolute value
                norm = np.maximum(norm, 1e-6)  # Clamp to a minimum value of 1e-6
                a /= norm
                b /= norm

                # Stack line parameters into a tensor and normalize to match au + bv = 1 form
                detected_lines = torch.from_numpy(np.stack((a, b), axis=-1)).to(
                    self.model.device
                )
                self.det_line_params = detected_lines
        else:
            self.det_line_params = None

        cTr = self._prev_cTr.clone()
        # joint_angles = self._prev_joint_angles.clone() # ignore the current joint angles

        cTr.requires_grad = True
        joint_angles.requires_grad = True

        optimizer = torch.optim.Adam([cTr, joint_angles], lr=self.lr)

        for _ in range(self.num_iters):
            optimizer.zero_grad()

            self.robot_mesh = self.robot_renderer.get_robot_mesh(joint_angles) 

            rendered_mask = self.model.render_single_robot_mask(cTr, self.robot_mesh, self.robot_renderer).squeeze(0) 

            mse = F.mse_loss(rendered_mask, mask)

            # cylinder loss
            if self.cylinder_loss:
                position = th.zeros(
                (1, 3), dtype=th.float32, device=self.model.device
                )  # (B, 3)
                # The direction of the cylinder is aligned along the z-axis
                direction = th.zeros((1, 3), dtype=th.float32, device=self.model.device)
                direction[:, 2] = 1.0  # Aligned along z-axis
                pose_matrix = self.model.cTr_to_pose_matrix(
                    cTr.unsqueeze(0)
                ).squeeze()

                radius = 0.0085 / 2  # adjust radius if needed
                # proj_position_2d, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
                # proj_norm_2d, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
                cylinder_val = self.cylinder_loss_single(
                    mask, position, direction, pose_matrix, radius
                )
            else:
                cylinder_val = 0.0

            # keypoint loss
            if self.kpts_loss:
                pose_matrix = self.model.cTr_to_pose_matrix(cTr.unsqueeze(0)).squeeze()

                R_list, t_list = lndFK(joint_angles)
                R_list = R_list.to(self.model.device)
                t_list = t_list.to(self.model.device)
                p_img1 = get_img_coords(
                    self.p_local1,
                    R_list[2],
                    t_list[2],
                    pose_matrix.to(joint_angles.dtype),
                    self.intr,
                )
                p_img2 = get_img_coords(
                    self.p_local2,
                    R_list[3],
                    t_list[3],
                    pose_matrix.to(joint_angles.dtype),
                    self.intr,
                )
                proj_keypoints = th.stack([p_img1, p_img2], dim=0)

                pts_val = self.keypoint_chamfer_loss(proj_keypoints, ref_keypoints)
            else:
                pts_val = 0.0

            loss = self.mse_weight * mse + self.pts_weight * pts_val + self.cylinder_weight * cylinder_val

            loss.backward()

            optimizer.step()

        # Save the current cTr and joint angles for the next iteration
        self._prev_cTr = cTr.detach()
        self._prev_joint_angles = joint_angles.detach()

        if visualization:
            overlay = self.overlay_mask(mask.detach(), rendered_mask.detach())
            return cTr.detach(), joint_angles.detach(), loss.item(), overlay
        else:
            return cTr.detach(), joint_angles.detach(), loss.item(), None


class DummyLogger(Logger):
    """
    A dummy logger that only maintains the best solution during the optimization.
    """
    def __init__(
        self, 
        searcher, 
        *, 
        interval: int = 1, 
        after_first_step: bool = False, ):
        # Call the super constructor
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

        self.best_solution, self.best_eval = None, float('inf')

    def __call__(self, status: dict):
        # Update best value and evaluation
        if status["pop_best_eval"] < self.best_eval:
            self.best_solution = status["pop_best"].values.clone()
            self.best_eval = status["pop_best_eval"]


class PoseEstimationProblem(Problem):
    def __init__(
        self, model, robot_renderer, ref_mask, intr, p_local1, p_local2
    ):
        super().__init__(
            objective_sense="min",
            solution_length=10, 
            device=model.device,
            initial_bounds=(
                th.tensor([-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi], device=model.device),
                th.tensor([math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], device=model.device),
            )
        )

        self.model = model
        self.robot_renderer = robot_renderer
        self.ref_mask = ref_mask

        self.intr = intr  
        self.p_local1 = p_local1  
        self.p_local2 = p_local2

        self.fx, self.fy, self.px, self.py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()
        
        self.ref_keypoints = None
        self.det_line_params = None

        self.kpts_loss = USE_PTS_LOSS
        self.cylinder_loss = USE_CYD_LOSS

        self.mse_weight = MSE_WEIGHT  # weight for the MSE loss
        self.pts_weight = PTS_WEIGHT  # weight for the keypoint loss
        self.cylinder_weight = CYD_WEIGHT  # weight for the cylinder loss
    
    # def _fill(self, values: torch.Tensor):
    #     raise NotImplementedError("Must initialize the problem with a solution.")

    def update_problem(
        self, ref_mask, ref_keypoints, det_line_params, joint_angles
    ):
        self.ref_mask = ref_mask
        self.ref_keypoints = ref_keypoints
        self.det_line_params = det_line_params
        self.joint_angles = joint_angles

    def cylinder_loss_batch(
        self, position, direction, pose_matrix, radius
    ):
        if self.det_line_params is None:
            print("No detected lines available for cylinder loss calculation.")
            return 0.0

        # get the projected cylinder lines parameters
        ref_mask = self.ref_mask
        intr = self.intr

        _, cam_pts_3d_position = transform_points_b(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points_b(direction, pose_matrix, intr)
        cam_pts_3d_norm = torch.nn.functional.normalize(
            cam_pts_3d_norm
        )  # NORMALIZE !!!!!!

        # print(f"checking shape of cylinder input: {cam_pts_3d_position.shape, cam_pts_3d_norm.shape}")  both [B,3]
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, self.fx, self.fy, self.px, self.py
        )  # [B,2], [B,2]
        projected_lines = torch.stack((e_1, e_2), dim=1)  # [B, 2, 2]

        detected_lines = self.det_line_params
        detected_lines = detected_lines.unsqueeze(0).expand(
            position.shape[0], detected_lines.shape[0], detected_lines.shape[1]
        )  # [B, 2, 2]
        cylinder_loss = cylinder_loss_params(detected_lines, projected_lines)  # [B]

        return cylinder_loss

    def _evaluate_batch(self, solutions: SolutionBatch):
        values = solutions.values.clone()
        cTr_batch = values[:, :6]  # shape (B, 6)
        joint_angles = values[:, 6:]  # shape (B, 4)
        B = cTr_batch.shape[0]
        self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
            B, self.ref_mask.shape[0], self.ref_mask.shape[1]
        )

        verts, faces = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles)

        pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast(
            cTr_batch, verts, faces, self.robot_renderer
        )  # shape (B,H,W)

        mse = F.mse_loss(
            pred_masks_b,
            self.ref_mask_b,
            reduction="none",
        ).mean(dim=(1, 2))

        if self.kpts_loss:
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch)  # [B, 4, 4]

            R_list, t_list = batch_lndFK(joint_angles)
            R_list = R_list.to(self.model.device) # [B, 4, 3, 3]
            t_list = t_list.to(self.model.device) # [B, 4, 3]
            
            p_img1 = get_img_coords_batch(
                self.p_local1,
                R_list[:,2,...],
                t_list[:,2,...],
                pose_matrix_b.squeeze().to(joint_angles.dtype),
                self.intr,
            )
            p_img2 = get_img_coords_batch(
                self.p_local2,
                R_list[:,3,...],
                t_list[:,3,...],
                pose_matrix_b.squeeze().to(joint_angles.dtype),
                self.intr,
            )
            # They are both B, 2

            proj_pts = torch.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]

            ref_kps_2d = self.ref_keypoints.unsqueeze(0).expand(B, -1, -1)  
            pts_val = keypoint_loss_batch(proj_pts, ref_kps_2d)  # [B]

        else:
            pts_val = 0.0

        if self.cylinder_loss:
            position = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction[:, 2] = 1.0
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch).squeeze(0)  # shape(B, 4, 4)
            radius = 0.0085 / 2

            # We'll do a small custom function "cylinder_loss_single" that returns (loss, angle).
            cylinder_val = self.cylinder_loss_batch(
                position,
                direction,
                pose_matrix_b,
                radius,
            )  

        else:
            cylinder_val = 0.0

        loss = self.mse_weight * mse + self.pts_weight * pts_val + self.cylinder_weight * cylinder_val

        solutions.set_evals(loss)


class SimplePoseEstimationProblem(PoseEstimationProblem):
    """
    Assuming the optimal joint angles are known, this problem only optimizes the 6D pose (cTr).
    """
    def __init__(
        self, model, robot_renderer, ref_mask, intr, p_local1, p_local2, joint_angles=None
    ):
        super().__init__(model, robot_renderer, ref_mask, intr, p_local1, p_local2)

        self.joint_angles = joint_angles

    def update_problem(self, ref_mask, ref_keypoints, det_line_params, joint_angles):
        """
        Update the problem with the reference mask, keypoints, detected line parameters, and joint angles.
        """
        self.ref_mask = ref_mask
        self.ref_keypoints = ref_keypoints
        self.det_line_params = det_line_params
        self.joint_angles = joint_angles

        # Compute the robot mesh once and store it since the joint angles are fixed.
        self.verts, self.faces = self.robot_renderer.get_robot_verts_and_faces(joint_angles)

        # Get R and t lists for the fixed joint angles
        R_list, t_list = lndFK(joint_angles)
        self.R_list = R_list.to(self.model.device)  # [4, 3, 3]
        self.t_list = t_list.to(self.model.device)  # [4, 3]

    def _evaluate_batch(self, solutions: SolutionBatch):
        values = solutions.values.clone()
        cTr_batch = values[:, :6]  # shape (B, 6)
        B = cTr_batch.shape[0]
        self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
            B, self.ref_mask.shape[0], self.ref_mask.shape[1]
        )

        pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast(
            cTr_batch, self.verts, self.faces, self.robot_renderer
        )  # shape (B,H,W)

        mse = F.mse_loss(
            pred_masks_b,
            self.ref_mask_b,
            reduction="none",
        ).mean(dim=(1, 2))

        if self.kpts_loss:
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch)  # [B, 4, 4]

            R_list, t_list = self.R_list, self.t_list
            
            p_img1 = get_img_coords_batch(
                self.p_local1,
                R_list[2],
                t_list[2],
                pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                self.intr,
            )
            p_img2 = get_img_coords_batch(
                self.p_local2,
                R_list[3],
                t_list[3],
                pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                self.intr,
            )
            # They are both B, 2

            proj_pts = torch.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]

            ref_kps_2d = self.ref_keypoints.unsqueeze(0).expand(B, -1, -1)  
            pts_val = keypoint_loss_batch(proj_pts, ref_kps_2d)  # [B]

        else:
            pts_val = 0.0

        if self.cylinder_loss:
            position = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction[:, 2] = 1.0
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch).squeeze(0)  # shape(B, 4, 4)
            radius = 0.0085 / 2

            # We'll do a small custom function "cylinder_loss_single" that returns (loss, angle).
            cylinder_val = self.cylinder_loss_batch(
                position,
                direction,
                pose_matrix_b,
                radius,
            )  

        else:
            cylinder_val = 0.0

        loss = self.mse_weight * mse + self.pts_weight * pts_val + self.cylinder_weight * cylinder_val

        solutions.set_evals(loss)


class EvoTracker(Tracker):
    def __init__(
        self, model, robot_renderer, init_cTr, init_joint_angles, 
        num_iters=5, intr=None, p_local1=None, p_local2=None, 
        stdev_init=5e-3, use_SNES=False, optimize_joint_angles=False
    ):
        super().__init__(model, robot_renderer, init_cTr, init_joint_angles, num_iters, intr, p_local1, p_local2)

        self.stdev_init = stdev_init  # Initial standard deviation for the optimization

        if not self.model.args.use_nvdiffrast or self.model.use_antialiasing:
            print("[Use NvDiffRast without antialiasing for black box optimization.]")
            self.model.args.use_nvdiffrast = True  # use NvDiffRast renderer
            self.model.use_antialiasing = False # do not use antialiasing as gradients are not needed

        if optimize_joint_angles:
            self.problem = PoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2)
        else:
            self.problem = SimplePoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2)  # Problem will be set in track method
            self.stdev_init = self.stdev_init[:6] if isinstance(self.stdev_init, torch.Tensor) else self.stdev_init

        self.optimizer = SNES if use_SNES else XNES

    def get_cylinder(self, mask):
        ref_mask_np = mask.detach().cpu().numpy()
        longest_lines = detect_lines(ref_mask_np, output=True)
        longest_lines = np.array(longest_lines, dtype=np.float64)

        if longest_lines.shape[0] < 2:
            # Force skip cylinder or fallback
            print(
                "WARNING: Not enough lines found by Hough transform. Skipping cylinder loss."
            )
            # You can set self.det_line_params to None or some fallback
            ret = None

        else:
            # print(f"debugging the longest lines {longest_lines}")
            x1 = longest_lines[:, 0]
            y1 = longest_lines[:, 1]
            x2 = longest_lines[:, 2]
            y2 = longest_lines[:, 3]
            # print(f"debugging the end points x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            # Calculate line parameters (a, b, c) for detected lines
            a = y2 - y1
            b = x1 - x2
            c = x1 * y2 - x2 * y1  # Determinant for the line equation

            # Normalize to match the form au + bv = 1
            # norm = c + 1e-6  # Ensure no division by zero
            norm = np.abs(c)  # Compute the absolute value
            norm = np.maximum(norm, 1e-6)  # Clamp to a minimum value of 1e-6
            a /= norm
            b /= norm

            # Stack line parameters into a tensor and normalize to match au + bv = 1 form
            detected_lines = torch.from_numpy(np.stack((a, b), axis=-1)).to(
                self.model.device
            )
            ret = detected_lines

        return ret

    @torch.no_grad()
    def track(self, mask, ref_img, joint_angles, visualization=False):
        # Update the optimization problem
        ref_mask = mask.to(self.model.device)
        ref_keypoints = None
        if self.problem.kpts_loss:
            ref_keypoints = get_reference_keypoints_auto(ref_img_path=None, ref_img=ref_img, num_keypoints=2)
            ref_keypoints = torch.tensor(ref_keypoints).squeeze().float().cuda()
        det_line_params = self.get_cylinder(mask) if self.problem.cylinder_loss else None
        self.problem.update_problem(
            ref_mask, ref_keypoints, det_line_params, joint_angles
        )
        
        # Initialize the solution with the previous cTr and joint angles
        cTr = self._prev_cTr.clone()
        # joint_angles = self._prev_joint_angles.clone() # ignore the current joint angles
        center_init = torch.cat([cTr, joint_angles], dim=0)

        # Define the searcher and logger
        searcher = self.optimizer(
            problem=self.problem,
            stdev_init=self.stdev_init if self.optimizer is not CMAES else 1e-2,
            center_init=center_init,
            popsize=50,
        )
        logger = DummyLogger(searcher, interval=1, after_first_step=True)

        # Run the optimization
        searcher.run(self.num_iters)

        # Extract the best solution and evaluation from the logger
        best_solution = logger.best_solution
        cTr, joint_angles = best_solution[:6], best_solution[6:]
        loss = logger.best_eval

        # Update the previous cTr and joint angles
        self._prev_cTr = cTr.detach()
        self._prev_joint_angles = joint_angles.detach()

        if visualization:
            # Render the predicted mask for visualization
            robot_mesh = self.robot_renderer.get_robot_mesh(joint_angles)
            rendered_mask = self.model.render_single_robot_mask(cTr, robot_mesh, self.robot_renderer).squeeze(0)
            overlay = self.overlay_mask(mask.detach(), rendered_mask.detach())
            return cTr.detach(), joint_angles.detach(), loss, overlay
        else:
            return cTr.detach(), joint_angles.detach(), loss, None


