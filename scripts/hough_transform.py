import cv2
import numpy as np

mask_path = "./data/consecutive_prediction/0617/0/00000.png"

# Read image and display canny edges
img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (13, 13), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

