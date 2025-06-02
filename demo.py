import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from value_align import ValueHistogramAlign

image_path = '/data/Datasets/ReID/market1501/bounding_box_train/0002_c1s2_050821_02.jpg'

# show original image & cdf
original = Image.open(image_path).convert('RGB')
hsv_orig = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2HSV)
v_orig = hsv_orig[:, :, 2]
hist_orig, _ = np.histogram(v_orig.flatten(), bins=256, range=(0, 256), density=True)
cdf_orig = np.cumsum(hist_orig)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(original)
axes[0].axis('off')
axes[0].set_title("Original Image")

axes[1].fill_between(np.arange(256), cdf_orig, color='skyblue', alpha=0.6)
axes[1].plot(cdf_orig, color='blue')
axes[1].set_xlim(0, 255)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Value CDF (Original)")

plt.tight_layout()
plt.show()

# show value aligned image & cdf
mu_values = [0, 60, 125, 185, 255]
std_values = [30, 60, 90]

n_rows = len(std_values)
n_cols = len(mu_values)

fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4 * n_cols, 3 * n_rows))

for row, std in enumerate(std_values):
    for col, mean in enumerate(mu_values):
        vam = ValueHistogramAlign(mean=mean, std=std)
        aligned = vam(original)

        hsv = cv2.cvtColor(np.array(aligned), cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        hist, _ = np.histogram(v.flatten(), bins=256, range=(0, 256), density=True)
        cdf = np.cumsum(hist)
        img_ax = axes[row, col * 2]
        cdf_ax = axes[row, col * 2 + 1]

        img_ax.imshow(aligned)
        img_ax.axis('off')
        img_ax.set_title(f"μ={mean}, σ={std}", fontsize=10)

        cdf_ax.fill_between(np.arange(256), cdf, color='skyblue', alpha=0.6)
        cdf_ax.plot(cdf, color='blue')
        
        cdf_ax.set_xlim(0, 255)
        cdf_ax.set_ylim(0, 1.1)
        cdf_ax.set_title("Value CDF", fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.suptitle("VAM: Image and Brightness CDF (Value Channel)", fontsize=16)
plt.show()