import torch
from torchvision import transforms
import cv2

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
submodule_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'deep_hough_transform'))
sys.path.insert(0, submodule_dir)

from model.network import Net 


class DeepCylinderLoss:
    """
    Use the DHT model to generate a heatmap and compute the loss.
    """
    def __init__(
        self, model_dir="./deep_hough_transform/dht_r50_nkl_d97b97138.pth", mask=None, 
        numAngle=100, numRho=100, img_size=(480, 640), input_size=(400, 400)
    ):

        self.model = Net(numAngle=100, numRho=100, backbone='resnet50').cuda()

        if os.path.isfile(model_dir):
            checkpoint = torch.load(model_dir)
            if 'state_dict' in checkpoint.keys():
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("=> no pretrained model found at '{}'".format(model_dir))

        self.model.eval()

        self.numAngle, self.numRho = numAngle, numRho
        self.H, self.W = img_size
        self.input_size = input_size
        self.D = (input_size[0] ** 2 + input_size[1] ** 2) ** 0.5  # Diagonal of the resized image
        self.dtheta = torch.pi / self.numAngle
        self.drho = (self.D + 1) / (self.numRho - 1)

        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.update_heatmap(mask)

    def update_heatmap(self, mask):
        self.heatmap = self.model(self.transform(mask)).squeeze()
        # self.heatmap = torch.exp(self.heatmap / 50.) 

    def line2ends(self, lines):
        """
        Convert a line (ax + by = 1) to its endpoints in the image.
        Input:
            line: torch.Tensor of shape (B, 2) representing a batch of line coefficients (a, b).
        Output:
            endpoints: torch.Tensor of shape (B, 2, 2) representing the endpoints of the lines.
        """
        def trunc_toward_zero(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x >= 0, x.floor(), x.ceil()).to(torch.int32)

        a = lines[:, 0]
        b = lines[:, 1]
        H, W = self.H, self.W

        x1 = torch.zeros_like(a)
        x2 = torch.full_like(a, fill_value=W)
        y1 = torch.zeros_like(b)
        y2 = torch.full_like(b, fill_value=H)

        mask_vert = (b == 0.)
        if mask_vert.any():
            xv = 1 / a[mask_vert]
            x1[mask_vert] = xv
            x2[mask_vert] = xv
        
        y1[~mask_vert] = 1 / b[~mask_vert]
        y2[~mask_vert] = (1 - a[~mask_vert] * x2[~mask_vert]) / b[~mask_vert]

        x1, y1, x2, y2 = [torch.where(t < 0, torch.ceil(t), torch.floor(t)).to(torch.int32) for t in (x1, y1, x2, y2)]
        
        return x1, y1, x2, y2

    def line2hough(self, line):
        """
        Convert a line (ax + by = 1) to Hough space coordinates.
        Input:
            line: torch.Tensor of shape (B, 2) representing a batchc of line coefficients (a, b).
        Output:
            hough_coords: torch.Tensor of shape (B, 2) representing the Hough space coordinates (theta, r).
        Note. The range of theta is [0, pi) and r is [-sqrt(H^2 + W^2), sqrt(H^2 + W^2)] where H = W = 400.
        """
        # Convert line coefficients to end points and scale them to the input size
        x1, y1, x2, y2 = self.line2ends(line)
        x1, x2 = x1 * self.input_size[1] / self.W, x2 * self.input_size[1] / self.W
        y1, y2 = y1 * self.input_size[0] / self.H, y2 * self.input_size[0] / self.H
        x1, y1, x2, y2 = [t.to(torch.int32) for t in (x1, y1, x2, y2)]

        # Compute alpha
        theta = torch.atan2(y2 - y1, x2 - x1) 
        alpha = theta + torch.pi / 2 # alpha = theta + pi/2 in [0, pi)

        # Compute r
        r = torch.zeros_like(theta)
        x1c, y1c = x1 - self.input_size[1] / 2, y1 - self.input_size[0] / 2 # center the coordinates
        mask_vert = (theta == -torch.pi / 2) 
        if mask_vert.any():
            r[mask_vert] = x1c[mask_vert]  # For vertical lines, r = x - W/2
        k = torch.tan(theta[~mask_vert])  # slope of the line
        r[~mask_vert] = (y1c[~mask_vert] - k * x1c[~mask_vert]) / torch.sqrt(1 + k**2)  # For non-vertical lines, r = (y - k*x) / sqrt(1 + k^2)

        return torch.stack([alpha, r], dim=1)

    def hough2idx(self, hough_coords):
        """
        Convert Hough space coordinates to indices in the heatmap.
        Input:
            hough_coords: torch.Tensor of shape (B, 2) representing Hough space coordinates (theta, r).
        Output:
            idx: torch.Tensor of shape (B, 2) representing indices in the heatmap.
        """
        theta, r = hough_coords[:, 0], hough_coords[:, 1]
    
        theta_idx = (theta / self.dtheta).round()
        r_idx = (r / self.drho + self.numRho // 2).round()

        # theta_idx = torch.clamp(theta_idx, min=0, max=self.numAngle - 1)
        # r_idx = torch.clamp(r_idx, min=0, max=self.numRho - 1)

        return torch.stack([theta_idx, r_idx], dim=1)

    def __call__(self, projected_lines):
        """
        Evaluate a batch of projected line pairs and compute the loss.
        Input:
            projected_lines: torch.Tensor of shape (B, 2， 2) representing a batch of line pairs
        Output:
            loss: torch.Tensor of shape (B,) representing the computed loss.
        """
        # Concatenate the two batches of lines
        B = projected_lines.shape[0]
        projected_lines_1 = projected_lines[:, 0, :]
        projected_lines_2 = projected_lines[:, 1, :]
        lines = torch.cat([projected_lines_1, projected_lines_2], dim=0)

        # Convert lines to Hough coordinates
        hough_coords = self.line2hough(lines)
        idx = self.hough2idx(hough_coords)

        # Compute the loss based on the heatmap
        loss = torch.full(fill_value=-float('inf'), size=(2*B,), dtype=torch.float32, device=lines.device)
        mask_in_bounds = (idx[:, 0] >= 0) & (idx[:, 0] < self.numAngle) & (idx[:, 1] >= 0) & (idx[:, 1] < self.numRho)
        if mask_in_bounds.any():
            loss[mask_in_bounds] = self.heatmap[idx[mask_in_bounds, 0].long(), idx[mask_in_bounds, 1].long()]

        # Split the loss back into two batches
        loss_1, loss_2 = loss[:B], loss[B:]
        
        return loss_1 + loss_2

if __name__ == "__main__": 
    from matplotlib import pyplot as plt
    import time

    # Example usage
    model_dir = "./deep_hough_transform/dht_r50_nkl_d97b97138.pth"
    mask_dir = "./deep_hough_transform/data/DVRK/7.jpg"
    
    img = cv2.imread(mask_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    mask = transforms.ToTensor()(img).unsqueeze(0).cuda() 

    with torch.no_grad():
        DHT_loss = DeepCylinderLoss(model_dir=model_dir, mask=mask, img_size=mask.shape[2:])

        start_time = time.time()
        DHT_loss.update_heatmap(mask)
        end_time = time.time()

    print(f"Time taken to update heatmap: {end_time - start_time:.4f} seconds")

    # Coupute the hough coordinates for the largest line in the heatmap
    heatmap = DHT_loss.heatmap.squeeze().cpu().numpy()
    
    # Use Nelder-Mead optimization to find the line parameters
    from scipy.optimize import minimize
    def objective_function(params):
        a, b = params
        # Convert (a, b) to Hough coordinates
        hough_coords = DHT_loss.line2hough(torch.tensor([[a, b]]).cuda())
        idx = DHT_loss.hough2idx(hough_coords)
        if idx[0, 0] < 0 or idx[0, 0] >= DHT_loss.numAngle or idx[0, 1] < 0 or idx[0, 1] >= DHT_loss.numRho:
            return float('inf')
        return -100 * DHT_loss.heatmap[idx[0, 0].long(), idx[0, 1].long()].item()  # Negative for maximization

    # Correct coordinates: [(303, 0, 251, 399), (227, 0, 237, 399)]
    a = 0.00053764774973738
    b = 0.0055005500550055
    initial_guess = [0.0005, 0.005]  # Initial guess for (a, b)
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True})
    a_opt, b_opt = result.x
    print(f"Optimized line parameters: a = {a_opt}, b = {b_opt}")
    print(f"Function evaluations: {result.nfev}, Function value: {result.fun}")

    # Plot the original image with the detected line and plot the corresponding points on the heatmap
    lines = torch.tensor([[a_opt, b_opt]]).cuda()  # Use the optimized line parameters
    hough_coords = DHT_loss.line2hough(lines)
    idx = DHT_loss.hough2idx(hough_coords)
    print(lines, hough_coords, idx)

    # Plot the original image with the detected line
    x1, y1, x2, y2 = DHT_loss.line2ends(lines)
    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
    print(f"Line endpoints: ({x1}, {y1}), ({x2}, {y2})")
    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)

    # Plot the original image and the heatmap
    heatmap = DHT_loss.heatmap.squeeze().cpu().numpy()
    # heatmap = torch.exp(DHT_loss.heatmap / 50.).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Mask")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    # draw the line on the heatmap
    plt.plot(idx[0, 1].item(), idx[0, 0].item(), 'bo', markersize=5)  # Mark the max point
    plt.title("Heatmap")
    plt.colorbar()
    plt.show()