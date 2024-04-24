# test file
# import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
# from stereomideval.dataset import Dataset
import helper_classes as helper

# use the package
# import graphcuts as gc

# load Tsukuba img
tsukubaImageFd = os.path.join(os.getcwd(),"datasets", "Tsukuba")

left_image = cv2.imread(os.path.join(tsukubaImageFd, "imL.png"), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join(tsukubaImageFd, "imR.png"), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(os.path.join(tsukubaImageFd, "groundtruth.png"), cv2.IMREAD_GRAYSCALE)
# only use a small part of the image

h = 80
# w = 240
x = 0
y = 120
imgL = left_image [y:y+h, :]
imgR = right_image [y:y+h, :]
print('Start running Graph cut')
dispSize=16
OCCLUDED = 1<<30

## not mine
"""
disparity = gc.helper(imgL, imgR)#, dispSize=16) # this is correctL L, R
im = np.zeros(imgL.shape, dtype=np.uint8)
occluded = disparity == 1<<30
im[np.logical_not(occluded)] = disparity[np.logical_not(occluded)] * 255/dispSize

im = cv2.applyColorMap(im, cv2.COLORMAP_PARULA)
im[occluded] = np.array([0,0,0])
cv2.imwrite('disparity_map_gc.png', im)
cv2.imwrite('disparity_map_gc_groundtruth.png', ground_truth[y:y+h, :])

print('Done')
# stop here
"""
# time the graph cut algorithm
start = cv2.getTickCount()  # start time
GraphCut = helper.GCKZ(imgL, imgR, maxIterations=4, alphaRange=dispSize)
disparity_map = GraphCut.compute_disparity_map()
# disparity will be negative d 
end = cv2.getTickCount()  # end time
time = (end - start) / cv2.getTickFrequency()  # time in seconds    
print('Time: ', time)    

# occulusion mask
occlusion_mask = disparity_map == OCCLUDED
print(f"percentage of occlusion: {np.sum(occlusion_mask) / disparity_map.size * 100:.2f}%")
# replace occluded pixels with 0
disparity_map[occlusion_mask] = 0
# flip sign of disparity
disparity_map = -1 * disparity_map
# min max of disparity
print(f"min disparity: {np.min(disparity_map)}; max disparity: {np.max(disparity_map)}")
# normalize the disparity map to 8-bit
disparity_map_norm = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# transform the disparity map to color with good separation of colors
# disparity_map_norm = cv2.applyColorMap(disparity_map_norm, cv2.COLORMAP_PARULA) 
# gray scale just duplicate for 3 channels
disparity_map_norm = cv2.merge((disparity_map_norm, disparity_map_norm, disparity_map_norm))

# replace the occluded pixels with black/cyan, 
disparity_map_norm[occlusion_mask] = 255, 255, 0
# plot the disparity map using cv2
# cv2.imshow('Disparity Map', disparity_map_norm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save the disparity map as png
cv2.imwrite('disparity_map_norm_mine.png', disparity_map_norm)
ground_truth_norm = cv2.normalize(ground_truth[y:y+h, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# ground_truth_norm = cv2.applyColorMap(ground_truth_norm, cv2.COLORMAP_PLASMA)
ground_truth_norm = cv2.merge((ground_truth_norm, ground_truth_norm, ground_truth_norm))

cv2.imwrite('disparity_map_norm_groundTruth_crop.png', ground_truth_norm)

# ground truth with occlusion
ground_truth_norm[occlusion_mask] = 255, 255, 0
cv2.imwrite('disparity_map_norm_groundTruth_crop_occ.png', ground_truth_norm) 
print('Done')
# """
