import torch
import kornia


@torch.compile()
def enforce_quaternion_consistency(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Enforces the consistency of quaternion signs by aligning them with the first quaternion in the batch.
    Input: quaternions (B, 4) where B is the batch size.
    Output: quaternions with consistent signs.
    """
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Input quaternions must have shape (B, 4)")

    # Compute the sign consistency based on the first quaternion's vector part
    temp = quaternions[0, 1:].unsqueeze(0)  # Use the first quaternion's vector part for sign consistency
    signs = torch.sign((temp * quaternions[1:,1:]).sum(dim=-1)).squeeze()  # Compute dot product to determine sign consistency
    signs = torch.cat([torch.tensor([1.0], device=quaternions.device), signs])
    signs = signs.unsqueeze(1).to(quaternions.device)  # Reshape to (B, 1) for broadcasting

    # Apply the signs to the quaternions
    quaternions = quaternions * signs

    return quaternions


@torch.compile()
def mix_angle_to_axis_angle(mix_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts [alpha, beta, gamma] (in radians) with
      R = R_y(gamma) @ R_x(alpha) @ R_z(beta)
    into (axis, angle) representation of shape (B,3),
    where the 3-vector is axis * angle.
    """
    # unpack
    alpha = mix_angle[:, 0]
    beta  = mix_angle[:, 1]
    gamma = mix_angle[:, 2]

    # pre‑compute sines/cosines
    ca, sa = alpha.cos(), alpha.sin()
    cb, sb = beta.cos(),  beta.sin()
    cg, sg = gamma.cos(), gamma.sin()

    # build each [B,3,3]
    # R_x(alpha)
    R_x = torch.stack([
        torch.stack([torch.ones_like(ca),  torch.zeros_like(ca),  torch.zeros_like(ca)], dim=-1),
        torch.stack([torch.zeros_like(ca), ca,                   -sa], dim=-1),
        torch.stack([torch.zeros_like(ca), sa,                    ca], dim=-1),
    ], dim=-2)

    # R_z(beta)
    R_z = torch.stack([
        torch.stack([ cb,                  -sb,                   torch.zeros_like(cb)], dim=-1),
        torch.stack([ sb,                   cb,                   torch.zeros_like(cb)], dim=-1),
        torch.stack([torch.zeros_like(cb),  torch.zeros_like(cb), torch.ones_like(cb)], dim=-1),
    ], dim=-2)

    # R_y(gamma)
    R_y = torch.stack([
        torch.stack([ cg,                   torch.zeros_like(cg), sg], dim=-1),
        torch.stack([ torch.zeros_like(cg), torch.ones_like(cg),  torch.zeros_like(cg)], dim=-1),
        torch.stack([-sg,                   torch.zeros_like(cg), cg], dim=-1),
    ], dim=-2)

    # compose: R = R_y @ R_x @ R_z
    R = R_y @ (R_x @ R_z)  # (B,3,3)

    # convert to axis-angle (axis * angle)
    axis_angle = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R)
    # quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(R)  # (B,4)
    # quaternions = enforce_quaternion_consistency(quaternions)  # Ensure sign consistency
    # axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternions)

    return axis_angle  # (B,3)

@torch.compile()
def axis_angle_to_mix_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle vectors (B,3) to a mixed angle representation [alpha, beta, gamma] (B,3),
    where R = R_y(gamma) @ R_x(alpha) @ R_z(beta).

    Input:
        axis_angle: (B,3) torch.Tensor where vector = axis * angle
    Output:
        mix_angle: (B,3) torch.Tensor with angles [alpha, beta, gamma] in radians
    """
    R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)  # (B, 3, 3)

    # Extract relevant components
    r10 = R[:, 1, 0]
    r11 = R[:, 1, 1]
    r12 = R[:, 1, 2]
    r02 = R[:, 0, 2]
    r22 = R[:, 2, 2]

    # α = asin(-r12)
    alpha = torch.asin(torch.clamp(-r12, -1.0, 1.0))  # [B]

    # cos(α) for division safety
    cos_alpha = torch.cos(alpha)
    cos_alpha = torch.where(cos_alpha.abs() < 1e-6, torch.full_like(cos_alpha, 1e-6), cos_alpha)

    # β = atan2(r10 / cos(α), r11 / cos(α))
    beta = torch.atan2(r10 / cos_alpha, r11 / cos_alpha)

    # γ = atan2(r02 / cos(α), r22 / cos(α))
    gamma = torch.atan2(r02 / cos_alpha, r22 / cos_alpha)

    return torch.stack([alpha, beta, gamma], dim=-1)