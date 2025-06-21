import cv2
import time
import torch
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from diffcali.utils.ui_utils import *

def Shi_Tomasi_corner_detection(
    image, max_corners=2, quality_level=0.1, min_distance=5, block_size=15
):
    """
    Detects corners in the image using Shi-Tomasi corner detection method.
    Use PyTorch without OpenCV.
    
    Args:
        image (torch.Tensor): Input image in grayscale format (B, H, W)
        max_corners (int): Maximum number of corners to return.
        quality_level (float): Parameter characterizing the minimal accepted quality of image corners.
        min_distance (float): Minimum possible Euclidean distance between the returned corners.
        block_size (int): Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
    Returns:
        torch.Tensor: Detected corners in the image, shape (B, N, 2) where N is the number of detected corners.
    """
    B, H, W = image.shape
    device = image.device

    # Compute image gradients (Sobel filters)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    Ix = F.conv2d(image.unsqueeze(1), sobel_x, padding=1).squeeze(1)
    Iy = F.conv2d(image.unsqueeze(1), sobel_y, padding=1).squeeze(1)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Apply box filter to compute sums over block_size region
    kernel = torch.ones((1, 1, block_size, block_size), device=device) / (block_size * block_size)
    Ix2 = F.conv2d(Ix2.unsqueeze(1), kernel, padding=block_size // 2).squeeze(1)
    Iy2 = F.conv2d(Iy2.unsqueeze(1), kernel, padding=block_size // 2).squeeze(1)
    Ixy = F.conv2d(Ixy.unsqueeze(1), kernel, padding=block_size // 2).squeeze(1)

    # Compute minimum eigenvalue of 2x2 structure tensor
    trace = Ix2 + Iy2
    det = Ix2 * Iy2 - Ixy * Ixy
    eig_min = (trace - torch.sqrt(trace ** 2 - 4 * det)) / 2

    # Thresholding
    max_response = eig_min.amax(dim=(1, 2), keepdim=True)
    threshold = max_response * quality_level
    mask = eig_min >= threshold

    # Get candidate coordinates
    coords = torch.nonzero(mask, as_tuple=False)
    batches = coords[:, 0]
    ys = coords[:, 1].float()
    xs = coords[:, 2].float()
    responses = eig_min[batches, coords[:, 1], coords[:, 2]]

    # Sort by response
    sorted_indices = torch.argsort(responses, descending=True)
    coords_sorted = coords[sorted_indices]

    # Non-maximum suppression with min_distance
    selected = []
    used = torch.zeros_like(eig_min, dtype=torch.bool)
    for idx in coords_sorted:
        b, y, x = idx
        if not used[b, y, x]:
            selected.append([b.item(), y.item(), x.item()])
            y0 = max(y - min_distance, 0)
            y1 = min(y + min_distance + 1, H)
            x0 = max(x - min_distance, 0)
            x1 = min(x + min_distance + 1, W)
            used[b, y0:y1, x0:x1] = True
        if len(selected) >= max_corners * B:
            break

    # Organize per batch
    selected_tensor = torch.full((B, max_corners, 2), -1.0, device=device)
    counts = torch.zeros(B, dtype=torch.int)
    for b, y, x in selected:
        if counts[b] < max_corners:
            selected_tensor[b, counts[b]] = torch.tensor([x, y], device=device)
            counts[b] += 1

    return selected_tensor

fname1 = "./data/consistency_evaluation/medium/0/00010.jpg"
fname2 = "./data/consistency_evaluation/medium/1/00120.jpg"

img1 = cv2.imread(fname1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
img2 = cv2.imread(fname2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

img1 = torch.tensor(img1, dtype=torch.float32).cuda().unsqueeze(0)  # Add batch dimension
img2 = torch.tensor(img2, dtype=torch.float32).cuda().unsqueeze(0)  # Add batch dimension
print(img1.shape)

keypoints1 = Shi_Tomasi_corner_detection(
    img1, max_corners=2, quality_level=0.1, min_distance=5, block_size=15
)

start_time = time.time()
keypoints2 = Shi_Tomasi_corner_detection(
    img2, max_corners=2, quality_level=0.1, min_distance=5, block_size=15
)
end_time = time.time()

print(f"Keypoints detected in {(end_time - start_time)*1000:.2f} ms")


