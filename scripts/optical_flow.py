import cv2
import numpy as np
import time

mask_1_path = "./data/consecutive_prediction/0617/0/00000.png"
mask_2_path = "./data/consecutive_prediction/0617/5/00005.png"

# Read image and display canny edges
mask_1 = cv2.imread(mask_1_path, cv2.IMREAD_GRAYSCALE)
mask_2 = cv2.imread(mask_2_path, cv2.IMREAD_GRAYSCALE)
mask_1 = cv2.GaussianBlur(mask_1, (13, 13), 0)
mask_2 = cv2.GaussianBlur(mask_2, (13, 13), 0)


start_time = time.time()
for _ in range(10):
    flow = cv2.calcOpticalFlowFarneback(mask_1, mask_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
end_time = time.time()
# print(flow)
print(flow.shape)
print(f"Time taken for 10 iterations: {(end_time - start_time)/10} seconds")

# Display the flow
hsv = np.zeros_like(cv2.cvtColor(mask_1, cv2.COLOR_GRAY2BGR))
hsv[..., 1] = 255
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("Optical Flow", bgr)

# display the original images overlayed
mask_1_colored = cv2.cvtColor(mask_1, cv2.COLOR_GRAY2BGR)
mask_2_colored = cv2.cvtColor(mask_2, cv2.COLOR_GRAY2BGR)
bgr = cv2.addWeighted(mask_1_colored, 0.5, mask_2_colored, 0.5, 0)
bgr = cv2.addWeighted(bgr, 0.5, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.5, 0)

cv2.imshow("Overlayed Images", bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()