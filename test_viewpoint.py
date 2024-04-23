# test view point function

# import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from stereomideval.dataset import Dataset

# load Tsukuba img
tsukubaImageFd = os.path.join(os.getcwd(),"datasets", "tsukuba")

left_image = cv2.imread(os.path.join(tsukubaImageFd, "imL.png"), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join(tsukubaImageFd, "imR.png"), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(os.path.join(tsukubaImageFd, "groundtruth.png"), cv2.IMREAD_GRAYSCALE)