import torch
import numpy as np
import FastGeodis
import cv2
import time
import matplotlib.pyplot as plt
import skfmm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_weighting_mask(shape, center_weight=0.5, edge_weight=1.0):
    """
    Copied from your single-sample code: creates a weighting mask for the MSE.
    shape: (H,W)
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    max_distance = np.sqrt(center_y**2 + center_x**2)
    normalized_distance = distance / max_distance
    weights = edge_weight + (center_weight - edge_weight) * (
        1 - normalized_distance
    )
    weighting_mask = torch.from_numpy(weights).float().cuda()

    return weighting_mask


def compute_edge_aware_cost(image_gray, lambda_val=1.0):
    image_gray = image_gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    cost_map = 1 + lambda_val * gradient_mag
    return torch.from_numpy(cost_map).float().cuda()


def compute_distance_map(ref_mask):
    ref_mask_np = ref_mask.detach().cpu().numpy().astype(np.float32)
    distance_map = skfmm.distance(ref_mask_np == 0)
    distance_map[ref_mask_np == 1] = 0
    return torch.from_numpy(distance_map).float().to(ref_mask.device)


if __name__ == "__main__":
    # Read binary mask (as grayscale image)
    img_dir = "data/consecutive_prediction/0617/0/00000.png"
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    mask = 1 -  torch.from_numpy(img).cuda().float().unsqueeze(0).unsqueeze(0) / 255
    weighting_mask = compute_edge_aware_cost(img).unsqueeze(0).unsqueeze(0).cuda()

    v, lamb, iterations = 1e10, 0.0, 2

    # warm up the GPU
    ret = FastGeodis.generalised_geodesic2d(
        weighting_mask,
        mask,  
        v, 
        lamb,
        iterations
    )

    start_time = time.time()
    ret = FastGeodis.generalised_geodesic2d(
        weighting_mask,
        mask,  
        v, 
        lamb,
        iterations
    )
    end_time = time.time()
    print(f"Time taken for generalized geodesic transform: {end_time - start_time:.4f} seconds")

    # Compare with skfmm
    cv_img = cv2.imread(img_dir)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_fmm = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0
    mask_fmm = torch.Tensor(img_fmm).cuda()
    ret_fmm = compute_distance_map(mask_fmm)

    start_time_fmm = time.time()
    ret_fmm = compute_distance_map(mask_fmm)
    end_time_fmm = time.time()
    print(f"Time taken for skfmm distance transform: {end_time_fmm - start_time_fmm:.4f} seconds")
    print("Speedup:", (end_time_fmm - start_time_fmm) / (end_time - start_time))

    # Display the original mask and the transformed image using plt subfigures
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    ax[0].set_title('Original Mask')
    ax[0].axis('off')
    ax[1].imshow(weighting_mask.squeeze().cpu().numpy())
    ax[1].set_title('Weighting Mask')
    ax[1].axis('off')
    ax[2].imshow(ret.squeeze().cpu().numpy())
    ax[2].set_title('Euclidean Distance Transform')
    ax[2].axis('off')
    ax[3].imshow(ret_fmm.squeeze().cpu().numpy())
    ax[3].set_title('Fast Marching Method')
    ax[3].axis('off')
    plt.tight_layout()
    plt.show()

