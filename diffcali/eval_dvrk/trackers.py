import torch
import torch as th
import torch.nn.functional as F
import kornia
import FastGeodis
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
from diffcali.utils.dht_utils import DeepCylinderLoss, SmoothDeepCylinderLoss
from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
    unscented_mix_angle_to_axis_angle,
    find_local_quaternion_basis,
)
from diffcali.utils.cma_es import (
    CMAES_cus, 
    generate_sigma_normal,
    generate_low_discrepancy_normal,
)

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import SearchAlgorithm, SNES, XNES, CMAES
from evotorch.logging import Logger, StdOutLogger


torch.set_default_dtype(torch.float32)
# torch.autograd.set_detect_anomaly(True)

LOWER_BOUNDS = [-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi]
UPPER_BOUNDS = [math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi]


def convert_line_params_to_endpoints(a, b, image_width, image_height):
    """
    Convert line parameters (a, b) in the form au + bv = 1 to endpoints (x1, y1), (x2, y2).

    Args:
        a (float): Parameter 'a' from the line equation.
        b (float): Parameter 'b' from the line equation.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Endpoints (x1, y1), (x2, y2) for drawing the line.
    """
    if a < 0:
        # Normalize to ensure a is positive
        a = -a
        b = -b

    # Calculate endpoints by choosing boundary values for 'u'
    if b != 0:
        # Set u = 0 to find y-intercept
        x1 = 0
        y1 = int((1 - a * x1) / b)

        # Set u = image_width to find corresponding 'v'
        x2 = image_width
        y2 = int((1 - a * x2) / b)
    else:
        # Vertical line: set v based on boundaries
        y1 = 0
        y2 = image_height
        x1 = x2 = int(1 / a)

    return (x1, y1), (x2, y2)


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


class DummyLogger(Logger):
    """
    A dummy logger that only maintains the best solution during the optimization.
    """
    def __init__(
        self, 
        searcher, 
        *, 
        interval: int = 1, 
        after_first_step: bool = False
    ):
        # Call the super constructor
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

        self.best_solution, self.best_eval = None, float('inf')

    def __call__(self, status: dict):
        # Update best value and evaluation
        if status["pop_best_eval"] < self.best_eval:
            self.best_solution = status["pop_best"].values.clone()
            self.best_eval = status["pop_best_eval"]
            # print(f"New best solution found: {self.best_solution}, evaluation: {self.best_eval}")

        # self.best_solution = status["pop_best"].values.clone()
        # self.best_eval = status["pop_best_eval"]

        self._steps_count += 1


class PoseEstimationProblem(Problem):
    def __init__(
        self, model, robot_renderer, ref_mask, intr, p_local1, p_local2, stdev_init, args
    ):
        super().__init__(
            objective_sense="min",
            solution_length=11 if args.use_global_quaternion else 10, 
            device=model.device,
            initial_bounds=(LOWER_BOUNDS, UPPER_BOUNDS) if args.use_global_quaternion else (LOWER_BOUNDS[:10], UPPER_BOUNDS[:10])
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

        self.args = args

        self.render_loss = self.args.use_render_loss
        self.kpts_loss = self.args.use_pts_loss
        self.cylinder_loss = self.args.use_cyd_loss
        self.dht_loss = self.args.use_dht_loss

        self.mse_weight = self.args.mse_weight  # weight for the MSE loss
        self.dist_weight = self.args.dist_weight  # weight for the distance loss
        self.app_weight = self.args.app_weight  # weight for the appearance loss
        self.pts_weight = self.args.pts_weight  # weight for the keypoint loss
        self.cylinder_weight = self.args.cyd_weight  # weight for the cylinder loss

        # Convert stdev_init to lengthscales
        if torch.is_tensor(stdev_init):
            self.lengthscales = stdev_init.clone().detach()
        else:
            self.lengthscales = stdev_init

        self.weighting_mask = None

        self.DHT_loss = DeepCylinderLoss(img_size=(480, 640)) if self.dht_loss else None
    
    # def _fill(self, values: torch.Tensor):
    #     raise NotImplementedError("Must initialize the problem with a solution.")

    def compute_weighting_mask(self, shape, center_weight=1.0, edge_weight=0.5):
        """
        Copied from your single-sample code: creates a weighting mask for the MSE.
        shape: (H,W)
        """
        if self.args.use_weighting_mask:
            h, w = shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            max_distance = np.sqrt(center_y**2 + center_x**2)
            normalized_distance = distance / max_distance
            weights = edge_weight + (center_weight - edge_weight) * (
                1 - normalized_distance
            )
            self.weighting_mask = torch.from_numpy(weights).float().to(self.model.device)

        else:
            self.weighting_mask = torch.ones(shape, dtype=torch.float32).to(self.model.device)

    def update_problem(
        self, ref_mask, ref_keypoints, det_line_params, joint_angles, cTr_init, stdev_init
    ):
        self.ref_mask = ref_mask
        self.ref_keypoints = ref_keypoints
        self.det_line_params = det_line_params
        self.joint_angles = joint_angles

        if self.weighting_mask is None:
            self.compute_weighting_mask(shape=self.ref_mask.shape)

        # print(stdev_init)
        if torch.is_tensor(stdev_init):
            self.lengthscales = stdev_init.clone().detach()
        else:
            self.lengthscales = stdev_init

        # Update the DHT loss with the reference mask
        if self.dht_loss:
            mask = ref_mask.repeat(3, 1, 1).unsqueeze(0)
            self.DHT_loss.update_heatmap(mask)

        # Compute distance map
        if self.dist_weight > 0.:
            mask = 1 - self.ref_mask.float()  # Invert the mask for the distance transform
            v, lamb, iterations = 1e10, 0.0, 2 # Use Euclidean distance transform only
            self.dist_map = FastGeodis.generalised_geodesic2d(
                self.weighting_mask.unsqueeze(0).unsqueeze(0),
                mask.unsqueeze(0).unsqueeze(0),
                v, 
                lamb,
                iterations
            ).squeeze()

        # import matplotlib.pyplot as plt
        # plt.imshow(self.dist_map.cpu().numpy())
        # plt.show()

        # Transform the initial rotation representations
        self.cTr_init = cTr_init
        self.pose_dim = 6

        if self.args.use_mix_angle:
            axis_angle = cTr_init[:3].unsqueeze(0)  # shape (1, 3)
            mix_angle = axis_angle_to_mix_angle(axis_angle)  # Convert to Euler angles
            self.cTr_init[:3] = mix_angle.squeeze(0)  # Replace the first 3 elements with Euler angles

        elif self.args.use_unscented_transform:
            # Use unscented transform to obtain the Gaussian distribution in axis-angle space
            axis_angle = cTr_init[:3].clone().unsqueeze(0)  # shape (1, 3)
            mix_angle = axis_angle_to_mix_angle(axis_angle).squeeze()  # Convert to Euler angles
            mean_aa, cov_aa = unscented_mix_angle_to_axis_angle(mix_angle, stdev_init[:3])

            # Determine the new orthogonal basis and lengthscale by eigenvalue decomposition
            L, Q = torch.linalg.eigh(cov_aa)
            self.cTr_init[:3] = mean_aa @ Q # transform the mean to the new basis
            self.lengthscales[:3] = torch.sqrt(torch.clamp(L, min=1e-9)) # avoid numerical instability
            self.Q = Q

        elif self.args.use_local_quaternion:
            axis_angle = cTr_init[:3].clone().unsqueeze(0)
            mix_angle = axis_angle_to_mix_angle(axis_angle).squeeze()
            ret = find_local_quaternion_basis(mix_angle, stdev_init[:3])
            self.q0, self.cTr_init[:3], self.basis_4D, self.lengthscales[:3] = ret
            # self.lengthscales[:3] *= 0.42

        elif self.args.use_global_quaternion:
            axis_angle = cTr_init[:3].unsqueeze(0)
            quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle).squeeze()
            quaternion = quaternion / quaternion.norm()
            self.cTr_init = torch.cat([quaternion, self.cTr_init[3:]])
            self.pose_dim = 7

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

    def compute_loss(self, raw_values):
        values = raw_values * self.lengthscales # scale the values by the lengthscales
            
        raw_cTr_batch = values[:, :self.pose_dim]  # shape (B, 6)
        joint_angles = values[:, self.pose_dim:]  # shape (B, 4)
        B = raw_cTr_batch.shape[0]

        if self.args.use_mix_angle:
            mix_angle_batch = raw_cTr_batch[:, :3]  # shape (B, 3)
            axis_angle_batch = mix_angle_to_axis_angle(mix_angle_batch)  # shape
            # cTr_batch[:, :3] = axis_angle_batch  # replace the first 3 elements with axis-angle
            cTr_batch = torch.cat(
                [axis_angle_batch, raw_cTr_batch[:, 3:]], dim=1
            ) 

        elif self.args.use_unscented_transform:
            transformed_angle_batch = raw_cTr_batch[:, :3]
            axis_angle_batch = transformed_angle_batch @ self.Q.T # transform back to the standard basis
            cTr_batch = torch.cat(
                [axis_angle_batch, raw_cTr_batch[:, 3:]], dim=1
            ) 

        elif self.args.use_local_quaternion:
            local_3D_batch = raw_cTr_batch[:, :3] # local 3D coordinatess
            q = local_3D_batch @ self.basis_4D.T + self.q0.unsqueeze(0) # standard 4D coordiantes
            axis_angle_batch = kornia.geometry.conversions.quaternion_to_axis_angle(q)
            cTr_batch = torch.cat(
                [axis_angle_batch, raw_cTr_batch[:, 3:]], dim=1
            ) 

        elif self.args.use_global_quaternion:
            quaternion_batch = raw_cTr_batch[:, :4]
            axis_angle_batch = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion_batch)
            cTr_batch = torch.cat(
                [axis_angle_batch, raw_cTr_batch[:, 4:]], dim=1
            ) 

        else:
            cTr_batch = raw_cTr_batch

        if self.render_loss:
            # Obtain the weighting mask
            weighting_mask = self.weighting_mask
            weighting_mask = weighting_mask.unsqueeze(0).expand(
                B, *[-1 for _ in weighting_mask.shape]
            )

            self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
                B, self.ref_mask.shape[0], self.ref_mask.shape[1]
            )

            verts, faces = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles)
            pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast(
                cTr_batch, verts, faces, self.robot_renderer
            )  # shape (B,H,W)

            mse = F.mse_loss(
                pred_masks_b * weighting_mask,
                self.ref_mask_b * weighting_mask,
                reduction="none",
            ).mean(dim=(1, 2))

            # Compute distance loss
            if self.dist_weight > 0.:
                dist_map_ref = self.dist_map.unsqueeze(0).expand(
                    B, *[-1 for _ in self.dist_map.shape]
                )  # [B, H, W]
                dist = torch.sum(
                    (pred_masks_b * weighting_mask) * (dist_map_ref * weighting_mask), 
                    dim=(1, 2)
                )
            else:
                dist = 0.0

            # Compute appearance loss
            if self.app_weight > 0.:
                sum_pred = torch.sum(pred_masks_b, dim=(1, 2))  # [B]
                sum_ref = torch.sum(self.ref_mask_b, dim=(1, 2))
                app = torch.abs(sum_pred - sum_ref) 
            else:
                app = 0.0

        else:
            mse, dist, app = 0.0, 0.0, 0.0

        if self.kpts_loss:
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch)  # [B, 4, 4]

            R_list, t_list = batch_lndFK(joint_angles)
            R_list = R_list.to(self.model.device) # [B, 4, 3, 3]
            t_list = t_list.to(self.model.device) # [B, 4, 3]
            
            p_img1 = get_img_coords_batch(
                self.p_local1,
                R_list[:,2,...],
                t_list[:,2,...],
                pose_matrix_b.to(joint_angles.dtype),
                self.intr,
            )
            p_img2 = get_img_coords_batch(
                self.p_local2,
                R_list[:,3,...],
                t_list[:,3,...],
                pose_matrix_b.to(joint_angles.dtype),
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

        elif self.dht_loss:
            position = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction[:, 2] = 1.0
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch).squeeze(0)  # shape(B, 4, 4)
            radius = 0.0085 / 2

            # get the projected cylinder lines parameters
            ref_mask = self.ref_mask
            intr = self.intr

            _, cam_pts_3d_position = transform_points_b(position, pose_matrix_b, intr)
            _, cam_pts_3d_norm = transform_points_b(direction, pose_matrix_b, intr)
            cam_pts_3d_norm = torch.nn.functional.normalize(
                cam_pts_3d_norm
            )  # NORMALIZE !!!!!!

            # print(f"checking shape of cylinder input: {cam_pts_3d_position.shape, cam_pts_3d_norm.shape}")  both [B,3]
            e_1, e_2 = projectCylinderTorch(
                cam_pts_3d_position, cam_pts_3d_norm, radius, self.fx, self.fy, self.px, self.py
            )  # [B,2], [B,2]
            projected_lines = torch.stack((e_1, e_2), dim=1)  # [B, 2, 2]
            
            cylinder_val = self.DHT_loss(projected_lines) # maximize in the heatmap

        else:
            cylinder_val = 0.0

        loss = (
            self.mse_weight * mse
            + self.dist_weight * dist
            + self.app_weight * app
            + self.pts_weight * pts_val 
            + self.cylinder_weight * cylinder_val
        )

        return loss

    def _evaluate_batch(self, batch: SolutionBatch) -> SolutionBatch:
        values = batch.values.clone()  # extract the values
        losses = self.compute_loss(values)  # shape (B,)
        batch.set_evals(losses)  # set the evaluations for the batch


class GradientDescentSearcher(SearchAlgorithm):
    def __init__(self, problem: Problem, stdev_init=1., center_init=None, popsize=None):
        SearchAlgorithm.__init__(
            self,
            problem=problem, 
            pop_best=self._get_pop_best,
            pop_best_eval=self._get_pop_best_eval
        )

        # Turn on antialiasing if using NvDiffRast renderer
        if self.problem.model.args.use_nvdiffrast and not self.problem.model.use_antialiasing:
            print("[Antialiasing is not enabled in the NvDiffRast renderer. This may lead to inaccurate gradients.]")
            print("[Enabling antialiasing for better gradients.]")
            self.problem.model.use_antialiasing = True # use antialiasing for better gradients

        # # Update center init by random sampling
        # with torch.no_grad():
        #     dim = self.problem.solution_length
        #     candidates = generate_sigma_normal(popsize, dim) + center_init.unsqueeze(0) # shape (popsize, dim)
        #     losses = self.problem.compute_loss(candidates)  # shape (popsize,)
        #     idx = torch.argmin(losses).item() # Get the index of the best solution
        #     center_init = candidates[idx]  # shape (dim,)

        # Initialize the variables and optimizer
        self.vars = center_init.detach().clone().unsqueeze(0).requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.vars], lr=1.)

        # Dummy data for the logger to process
        self.batch = self.problem.generate_batch(1)
        self._pop_best: Optional[Solution] = None

    def _get_pop_best(self):
        return self._pop_best

    def _get_pop_best_eval(self):
        return self._pop_best.evals[0].item()

    def _step(self):
        # Back-propagate the loss
        self.optimizer.zero_grad()

        loss = self.problem.compute_loss(self.vars).squeeze()

        loss.backward(retain_graph=True)

        self.optimizer.step()

        # Update dummy data
        self.batch.set_values(self.vars.detach().clone())
        self.batch.set_evals(loss.unsqueeze(0).detach().clone())
        self._pop_best = self.batch[0]


class NelderMeadSearcher(SearchAlgorithm):
    def __init__(
        self,
        problem: Problem,
        stdev_init: float = 1.0,
        center_init: torch.Tensor = None,
        popsize=None,
        *,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ):
        # Initialize base SearchAlgorithm with status getters
        SearchAlgorithm.__init__(
            self,
            problem=problem,
            pop_best=self._get_pop_best,
            pop_best_eval=self._get_pop_best_eval,
        )
        # Store Nelder-Mead coefficients
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.sigma = float(sigma)

        # Determine dimension from center_init
        if center_init is None:
            raise ValueError("center_init must be provided for Nelder-Mead search.")
        dim = center_init.shape[0]

        # # Initialize and evaluate simplex
        # simplex = center_init.clone().unsqueeze(0).repeat(dim+1, 1).cuda()
        # for i in range(dim):
        #     simplex[i+1, i] += 1. # Add offsets along each axis to form the simplex vertices
        # losses = self.problem.compute_loss(simplex)
        candidates = generate_sigma_normal(popsize, dim) + center_init.unsqueeze(0)
        losses = self.problem.compute_loss(candidates)  # shape (popsize,)
        _, indices = torch.topk(losses, k=dim + 1, largest=False)  # Get the indices of the best dim+1 solutions
        simplex = candidates[indices]  # shape (dim + 1, dim)
        losses = losses[indices]  # shape (dim + 1,)

        # Track best solution so far
        self.simplex = self.problem.generate_batch(dim + 1)
        self.simplex.set_values(simplex)
        self.simplex.set_evals(losses)
        self._pop_best: Optional[Solution] = None

    def _get_pop_best(self):
        return self._pop_best

    def _get_pop_best_eval(self):
        return self._pop_best.evals[0].item()

    def _step(self):
        values = self.simplex.values.clone()  # shape (dim + 1, dim)
        losses = self.simplex.evals.clone().squeeze() # shape (dim + 1,)
        dim = values.shape[1]

        # Extract best, worst, and centroid of the simplex
        best_idx, worst_idx = torch.argmin(losses).item(), torch.argmax(losses).item()
        masked_losses = losses.clone()
        masked_losses[worst_idx] = -float("inf")  # exclude the worst
        second_worst_idx = torch.argmax(masked_losses).item()
        mask_worst = torch.ones(values.shape[0], dtype=bool, device=values.device)
        mask_worst[worst_idx] = False

        x_best = values[best_idx]
        x_worst = values[worst_idx]
        
        mask_worst = torch.ones(values.shape[0], dtype=bool, device=values.device)
        mask_worst[worst_idx] = False
        centroid =values[mask_worst].mean(dim=0)

        # Compute reflection, expansion, contraction and shrink
        x_reflect = centroid + self.alpha * (centroid - x_worst)
        x_exp = centroid + self.gamma * (x_reflect - centroid)
        x_cont_outside = centroid + self.rho * (x_reflect - centroid)  # contraction outside
        x_cont_inside = centroid + self.rho * (x_worst - centroid) # contraction inside
        simplex_shrink = values + self.sigma * (x_best - values) # shrink the simplex towards the best solution

        # Evaluate losses for reflection, expansion, contractiono and shrink at once
        x_cat = torch.cat(
            [x_reflect.unsqueeze(0), x_exp.unsqueeze(0), x_cont_outside.unsqueeze(0), x_cont_inside.unsqueeze(0), simplex_shrink], 
            dim=0
        )
        new_losses = self.problem.compute_loss(x_cat)  # shape (dim + 5,)

        # Assign the losses to the corresponding solutions
        loss_reflect = new_losses[0]
        loss_exp = new_losses[1]
        loss_cont_outside = new_losses[2]
        loss_cont_inside = new_losses[3]
        loss_shrink = new_losses[4:]

        shrink_needed = False

        if loss_reflect < losses[second_worst_idx]:
            if loss_reflect >= losses[best_idx]:
                new_point = x_reflect
                new_loss = loss_reflect
            else:
                if loss_exp < loss_reflect:
                    new_point = x_exp
                    new_loss = loss_exp
                else:
                    new_point = x_reflect
                    new_loss = loss_reflect
            values[worst_idx] = new_point
            losses[worst_idx] = new_loss

        else:
            if loss_reflect < losses[worst_idx]:
                if loss_cont_outside < loss_reflect:
                    values[worst_idx] = x_cont_outside
                    losses[worst_idx] = loss_cont_outside
                else:
                    shrink_needed = True
            else:
                if loss_cont_inside < losses[worst_idx]:
                    values[worst_idx] = x_cont_inside
                    losses[worst_idx] = loss_cont_inside
                else:
                    shrink_needed = True

        if shrink_needed:
            values = simplex_shrink
            losses = loss_shrink

        self.simplex.set_values(values.clone())
        self.simplex.set_evals(losses.clone())

        best_idx = torch.argmin(losses).item()
        self._pop_best = self.simplex[best_idx]

        # # Combine with the current simplex and determine the top dim+1 solutions
        # all_values = torch.cat([values, x_cat], dim=0)  # shape (2*dim + 6, dim)
        # all_losses = torch.cat([losses, new_losses], dim=0)  # shape (2*dim + 6,)
        # # sorted_indices = torch.argsort(all_losses)  # sort by losses
        # # updated_simplex_values = all_values[sorted_indices][:dim + 1]  # take the top dim+1 solutions
        # # updated_losses = all_losses[sorted_indices][:dim + 1]  # take the top dim+1 losses
        # updated_losses, sorted_indices = torch.topk(all_losses, k=dim + 1, largest=False)
        # updated_simplex_values = all_values[sorted_indices]
        
        # # Update the SolutionBatch and track best solution
        # self.simplex.set_values(updated_simplex_values.clone())
        # self.simplex.set_evals(updated_losses.clone())

        # # Store the new best solution
        # best_idx = torch.argmin(updated_losses).item()
        # self._pop_best = self.simplex[best_idx]


class RandomSearcher(SearchAlgorithm):
    def __init__(self, 
        problem: Problem, 
        stdev_init = 1., 
        center_init = None, 
        popsize = None,
        *,
        L: float = 1.,
        cont_coef: float = 0.5,
        exp_coef: float = 2.,
        L_max = 1.5,
        L_min = 0.1,
        tau_succ = 3,
        tau_fail = 3,
    ):
        SearchAlgorithm.__init__(
            self,
            problem=problem, 
            pop_best=self._get_pop_best,
            pop_best_eval=self._get_pop_best_eval
        )

        self.L, self.cont_coef, self.exp_coef = L, cont_coef, exp_coef
        self.L_max, self.L_min = L_max, L_min
        self.tau_succ, self.tau_fail = tau_succ, tau_fail
        self.cnt_succ, self.cnt_fail = 0, 0

        self.center = center_init.clone().unsqueeze(0)
        self.center_loss = float('inf')

        # Dummy data for the logger to process
        self.popsize, self.dim = popsize, problem.solution_length
        self.batch = self.problem.generate_batch(popsize)
        self._pop_best: Optional[Solution] = None

    def _get_pop_best(self):
        return self._pop_best

    def _get_pop_best_eval(self):
        return self._pop_best.evals[0].item()

    def _step(self):
        # Generate and evaluate candidates
        # candidates = generate_low_discrepancy_normal(self.popsize, self.dim, ratio=0.7) * self.L + self.center
        # candidates = generate_sigma_normal(self.popsize, self.dim) * self.L + self.center
        sobol_points = torch.quasirandom.SobolEngine(
            self.dim, scramble=True
        ).draw(self.popsize).cuda()  # shape (popsize, dim)
        candidates = self.L * (sobol_points - 0.5) + self.center  # scale and shift to the center
        losses = self.problem.compute_loss(candidates)  # shape (popsize,)

        # Update the center
        best_idx = torch.argmin(losses).item()
        if losses[best_idx] <= self.center_loss:
            self.center = candidates[best_idx].unsqueeze(0)
            self.center_loss = losses[best_idx].item()
            self.cnt_succ += 1
            self.cnt_fail = 0

        else:
            self.cnt_succ = 0
            self.cnt_fail += 1

        if self.cnt_succ >= self.tau_succ:
            self.L = min(self.L * self.exp_coef, self.L_max)  # expand L
            self.cnt_succ = 0

        elif self.cnt_fail >= self.tau_fail:
            self.L = max(self.L * self.cont_coef, self.L_min)  # shrink L
            self.cnt_fail = 0

        # print(self.L)

        # self.L *= 0.8

        # Update the best solution
        self.batch.set_values(candidates.clone())
        self.batch.set_evals(losses.clone())
        self._pop_best = self.batch[best_idx]


class Tracker:
    def __init__(
        self, model, robot_renderer, init_cTr, init_joint_angles, 
        num_iters=5, intr=None, p_local1=None, p_local2=None, 
        stdev_init=1., searcher="CMA-ES", optimize_joint_angles=True, args=None
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

        self.args = args

        self.fx, self.fy, self.px, self.py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()

        self.stdev_init = stdev_init  # Initial standard deviation for the optimization

        if not self.model.args.use_nvdiffrast or self.model.use_antialiasing:
            print("[Use NvDiffRast without antialiasing for black box optimization.]")
            self.model.args.use_nvdiffrast = True  # use NvDiffRast renderer
            self.model.use_antialiasing = False # do not use antialiasing as gradients are not needed

        # if optimize_joint_angles:
        #     self.problem = PoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2, self.stdev_init)
        # else:
        #     self.stdev_init = self.stdev_init[:6] if isinstance(self.stdev_init, torch.Tensor) else self.stdev_init
        #     self.problem = SimplePoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2, self.stdev_init)  # Problem will be set in track method

        assert optimize_joint_angles, "Currently only optimizing joint angles is supported."
        self.problem = PoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2, self.stdev_init, args)

        optimizer_dict = {
            "CMA-ES": CMAES_cus, # customized CMA-ES implementation
            "XNES": XNES,
            "SNES": SNES,
            "Gradient": GradientDescentSearcher,
            "NelderMead": NelderMeadSearcher,
            "RandomSearch": RandomSearcher,
        }
        self.optimizer = optimizer_dict[searcher]

        # Transform the intial cTr to Euler angle if required
        if self.args.use_mix_angle:
            print("[Using transformed angle space for optimization.]")
        elif self.args.use_unscented_transform:
            print("[Using orthogonally transformed axis-angle space for optimization.]")
        elif self.args.use_local_quaternion:
            print("[Using local 3D parameterizations of quaternions for optimization.]")
        elif self.args.use_global_quaternion:
            print("[Using 4D quaternions for optimization.]")
        else:
            print("[Using axis-angle space for optimization.]")

    def overlay_mask(self, ref_mask, pred_mask, ref_pts=None, proj_pts=None, ref_lines=None, proj_lines=None):
        """
        Overlay the predicted mask on the reference mask for visualization.
        """
        # Convert masks to grayscale images
        ref_mask = ref_mask.float().cpu().numpy()
        pred_mask = pred_mask.float().cpu().numpy()
        # ref_mask = (ref_mask*255).astype(np.uint8)
        # pred_mask = (pred_mask*255).astype(np.uint8)

        w, h = ref_mask.shape[1], ref_mask.shape[0]
        
        # Create a color overlay
        # rendered_color = np.stack([0.5 * pred_mask, 0.8 * pred_mask, np.zeros_like(pred_mask)], axis=-1) # Light blue for rendered mask
        # ref_color = np.stack([ref_mask, 0.6 * ref_mask, 0.1 * ref_mask], axis=-1)  # Orange for reference mask
        # overlay = rendered_color * 0.5 + ref_color * 0.5
        # # overlay = np.clip(overlay, 0, 1)
        # overlay = (overlay * 255).astype(np.uint8)
        # overlay = np.clip(overlay, 0, 255)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Create an empty overlay image
        overlay[..., 0] = (ref_mask * 255).astype(np.uint8)  # Blue channel 
        overlay[..., 2] = (pred_mask * 255).astype(np.uint8)  # Green channel

        if ref_pts != None:
            center_ref_pt = th.mean(ref_pts, dim=0)
            for ref_pt in ref_pts:
                u_ref, v_ref = int(ref_pt[0]), int(ref_pt[1])
                cv2.circle(
                    overlay,
                    (u_ref, v_ref),
                    radius=5,
                    color=(255, 0.6 * 255, 0.1 * 255),
                    thickness=-1,
                )  # Green

            u_ref, v_ref = int(center_ref_pt[0]), int(center_ref_pt[1])
            cv2.circle(
                overlay,
                (u_ref, v_ref),
                radius=5,
                color=(255, 0.6 * 255, 0.1 * 255),
                thickness=-1,
            )  # Green  BGR
            # Draw projected keypoints in red

        if proj_pts is not None:
            center_proj_pt = th.mean(proj_pts, dim=0)
            for proj_pt in proj_pts.squeeze():
                # print(f"debugging the project pts {proj_pts}")
                u_proj, v_proj = int(proj_pt[0].item()), int(proj_pt[1].item())
                cv2.circle(
                    overlay, (u_proj, v_proj), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
                )  # Red

            u_ref, v_ref = int(center_proj_pt[0]), int(center_proj_pt[1])
            cv2.circle(
                overlay, (u_ref, v_ref), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
            )  # Green

        # Draw detected lines in blue (convert from line parameters to endpoints)
        if ref_lines is not None:
            for line_params in ref_lines:
                a, b = line_params
                (x1, y1), (x2, y2) = convert_line_params_to_endpoints(a.item(), b.item(), w, h)
                cv2.line(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (255, 0.6 * 255, 0.1 * 255),
                    thickness=2,
                ) 

        # Draw projected lines in cyan (convert from line parameters to endpoints)
        if proj_lines is not None:
            for line_params in proj_lines:
                a, b = line_params
                (x1, y1), (x2, y2) = convert_line_params_to_endpoints(a, b, w, h)
                cv2.line(
                    overlay, (x1, y1), (x2, y2), (255*0.1, 255*0.6, 255), thickness=2
                ) 


        return overlay

    def track(self, mask, joint_angles, ref_keypoints, det_line_params, visualization=False):
        # Update the optimization problem
        ref_mask = mask.to(self.model.device)
        self.problem.update_problem(
            ref_mask, ref_keypoints, det_line_params, joint_angles, self._prev_cTr.clone(), self.stdev_init
        )

        # Initialize the solution with the previous cTr and joint angles
        cTr = self.problem.cTr_init
        joint_angles = self._prev_joint_angles if self.args.use_prev_joint_angles else joint_angles

        # joint_angles = self._prev_joint_angles.clone() # ignore the current joint angles
        center_init = torch.cat([cTr, joint_angles], dim=0)

        # Define the searcher and logger
        searcher = self.optimizer(
            problem=self.problem,
            stdev_init=1.,
            center_init=center_init / self.problem.lengthscales,
            popsize=self.args.popsize,
        )
        logger = DummyLogger(searcher, interval=1, after_first_step=False)
        # logger2 = StdOutLogger(searcher, interval=1, after_first_step=False)

        searcher.run(self.num_iters)

        # Extract the best solution and evaluation from the logger
        best_solution = logger.best_solution * self.problem.lengthscales
        cTr, joint_angles = best_solution[:self.problem.pose_dim], best_solution[self.problem.pose_dim:]
        loss = logger.best_eval

        # Convert the transformed rotation representations back to the axis-angle space
        if self.args.use_mix_angle:
            mix_angle = cTr[:3].unsqueeze(0)
            axis_angle = mix_angle_to_axis_angle(mix_angle) # Convert to axis-angle
            cTr = torch.cat([axis_angle.squeeze(), cTr[3:]], dim=0)       # Replace the first 3 elements with axis-angle

        elif self.args.use_unscented_transform:
            transformed_angle = cTr[:3].unsqueeze(0)
            axis_angle = transformed_angle @ self.problem.Q.T # transform back to the standard basis
            cTr = torch.cat([axis_angle.squeeze(), cTr[3:]], dim=0)

        elif self.args.use_local_quaternion:
            local_3D = cTr[:3].unsqueeze(0) # local 3D coordinates
            q = local_3D @ self.problem.basis_4D.T + self.problem.q0.unsqueeze(0) # standard 4D coordiantes
            axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(q)
            cTr = torch.cat([axis_angle.squeeze(), cTr[3:]], dim=0)      

        elif self.args.use_global_quaternion:
            quaternion = cTr[:4].unsqueeze(0)
            axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
            cTr = torch.cat([axis_angle.squeeze(), cTr[4:]])

        # Update the previous cTr and joint angles
        self._prev_cTr = cTr.detach().clone()
        self._prev_joint_angles = joint_angles.detach().clone()

        if visualization:
            # Render the predicted mask for visualization
            robot_mesh = self.robot_renderer.get_robot_mesh(joint_angles)
            rendered_mask = self.model.render_single_robot_mask(cTr, robot_mesh, self.robot_renderer).squeeze(0)

            # Project keypoints
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

            # Project cylinders
            cTr_batch = cTr.unsqueeze(0)  # shape (1, 6)
            B = 1
            position = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction[:, 2] = 1.0
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_batch).squeeze(0)  # shape(B, 4, 4)
            radius = 0.0085 / 2
            intr = self.intr
            fx, fy, px, py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()

            _, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
            _, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
            cam_pts_3d_norm = th.nn.functional.normalize(cam_pts_3d_norm)
            e_1, e_2 = projectCylinderTorch(
                cam_pts_3d_position, cam_pts_3d_norm, radius, fx, fy, px, py
            )  # [B,2], [B,2]
            projected_lines = torch.stack((e_1, e_2), dim=1)  # [B, 2, 2]

            # print(det_line_params.shape, projected_lines.shape)

            # Plot the overlay mask
            overlay = self.overlay_mask(
                mask.detach(), 
                rendered_mask.detach(),
                ref_pts=ref_keypoints,
                proj_pts=proj_keypoints,
                ref_lines=det_line_params.squeeze() if det_line_params is not None else None,
                proj_lines=projected_lines.squeeze()
            )

            return cTr.detach(), joint_angles.detach(), loss, overlay
        else:
            return cTr.detach(), joint_angles.detach(), loss, None


