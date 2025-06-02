import torch
import numpy as np
from PIL import Image
import cv2
from scipy.stats import norm

class ValueHistogramAlign:
    def __init__(self, mean=125, std=60):
        self.mean = mean
        self.std = std
        self.L = 256

        self.ref_values = np.arange(self.L)
        self.ref_pdf = norm.pdf(self.ref_values, loc=mean, scale=std)
        self.ref_pdf /= self.ref_pdf.sum()
        self.ref_cdf = np.cumsum(self.ref_pdf)

    def __call__(self, img: Image.Image):
        img = np.array(img.convert('RGB'))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        v = hsv[:, :, 2]
        flat = v.flatten()
        hist, bins = np.histogram(flat, bins=self.L, range=(0, self.L), density=True)
        src_cdf = np.cumsum(hist)
        
        # Histogram Matching: G(v) = R(r) ⇒ find r = R^-1(G(v))
        mapping = np.interp(src_cdf, self.ref_cdf, self.ref_values)
        v_mapped = np.interp(v.flatten(), bins[:-1], mapping)
        hsv[:, :, 2] = v_mapped.reshape(v.shape).astype(np.uint8)

        aligned_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(aligned_rgb)