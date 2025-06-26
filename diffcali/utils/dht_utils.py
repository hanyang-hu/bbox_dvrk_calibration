import torch
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
    def __init__(self, model_dir="./deep_hough_transform/dht_r50_nkl_d97b97138.pth", mask=None):

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

        self.update_heatmap(mask)

    @torch.no_grad()
    def update_heatmap(self, mask):
        self.mask = mask
        self.heatmap = self.model(mask).squeeze()

if __name__ == "__main__": 
    from PIL import Image
    from matplotlib import pyplot as plt
    from torchvision import transforms
    import time

    # Example usage
    model_dir = "./deep_hough_transform/dht_r50_nkl_d97b97138.pth"
    mask_dir = "./deep_hough_transform/data/DVRK/7.jpg"
    img = Image.open(mask_dir).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    mask = transform(img).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    DHT_loss = DeepCylinderLoss(model_dir=model_dir, mask=mask)

    start_time = time.time()
    DHT_loss.update_heatmap(mask)
    end_time = time.time()

    print(f"Time taken to update heatmap: {end_time - start_time:.4f} seconds")

    # Plot the original image and the heatmap
    print(DHT_loss.heatmap.shape)
    heatmap = DHT_loss.heatmap.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Mask")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Heatmap")
    plt.colorbar()
    plt.show()
    import time
    from matplotlib import pyplot