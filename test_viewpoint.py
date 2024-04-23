# test view point function

# import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
# from stereomideval.dataset import Dataset
import helper_classes as helper

# load Tsukuba img
tsukubaImageFd = os.path.join(os.getcwd(),"datasets", "tsukuba")

left_image = cv2.imread(os.path.join(tsukubaImageFd, "imL.png"), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join(tsukubaImageFd, "imR.png"), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(os.path.join(tsukubaImageFd, "groundtruth.png"), cv2.IMREAD_GRAYSCALE)
left_original = cv2.imread(os.path.join(tsukubaImageFd, "imL.png"), cv2.IMREAD_COLOR)
helper.plot_3D_image(left_original, ground_truth, f'new_view_window_test_gt.png')

# compute disparity using window method\
window_s = 15
# max disparity to 16 multiples from ground_truth
ndisp = 16 # * math.ceil(np.max(ground_truth) / 16)
disparity_map = helper.window_disparity(left_image, right_image, window_size=window_s, min_disparity=0,
                                         max_disparity=ndisp, consensus_tol=5, normalize=False)

# simple method to shift camera position on the baseline
# viewpoint_shift = 0.9 # values <1.0 shift towards right, >1.0 shift towards left.
# new_image = helper.present_view(left_image, right_image, disparity_map, viewpoint_shift)
# # write to png file
# cv2.imwrite((f'new_view_window_test.png'), new_image)

helper.plot_3D_image(left_original, disparity_map, f'new_view_window_test.png')

print("done")