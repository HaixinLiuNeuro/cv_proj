# test file
# import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from stereomideval.dataset import Dataset

# use my own stereo matching
import helper_classes as helper


# load Tsukuba img
tsukubaImageFd = os.path.join(os.getcwd(),"datasets", "tsukuba")

left_image = cv2.imread(os.path.join(tsukubaImageFd, "imL.png"), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join(tsukubaImageFd, "imR.png"), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(os.path.join(tsukubaImageFd, "groundtruth.png"), cv2.IMREAD_GRAYSCALE)
# only use a small part of the image

h = 60
# w = 240
x = 0
y = 120
imgL = left_image[y:y+h, :]
imgR = right_image[y:y+h, :]


GraphCut = helper.GCKZ(left_image, right_image)
disparity_map = GraphCut.compute_disparity_map()

# replace occluded pixels with 0
disparity_map[disparity_map == 1<<30] = 0
# normalize the disparity map to 8-bit
disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# plot the disparity map using cv2
cv2.imshow('Disparity Map', disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the disparity map as png
cv2.imwrite('disparity_map.png', disparity_map)

