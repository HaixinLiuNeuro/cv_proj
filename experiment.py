"""
This is the main experiment script

"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import helper_classes as helper
from stereomideval.dataset import Dataset # for getting data


DATASET_FOLDER = os.path.join(os.getcwd(),"datasets") #Path to download datasets
DISPLAY_IMAGES = False


def test():
    print('Hello World')
    
def download_data():
    """
    Download Milddlebury's data using helper classes
    
    """
    helper.download_dataset(DATASET_FOLDER)
    print('Data downloaded successfully')
    
def window_method_demo():
    """
    use the classical image pair (1 pair) to demonstrate the window method, with window = 5, 15, 35    
    save the disparity maps as png, scale to 8-bit
    present result from a different view and save as png result
    
    """
    viewpoint_shift = 0.9
    eval_methods = ['avg_abs', 'rms', 'A50', 'A90', 'A95', 'A99', 'bad0.5', 'bad1.0', 'bad2.0']
    
    # Get list of scenes in Milddlebury's stereo training dataset and iterate through them
    for scene_info in Dataset.get_training_scene_list():
        scene_name=scene_info.scene_name
        dataset_type=scene_info.dataset_type
        scene_data = Dataset.load_scene_data(
            scene_name=scene_name,dataset_folder=DATASET_FOLDER,
            dataset_type=dataset_type,display_images=False)
        # get image pair and others
        left_image = scene_data.left_image
        right_image = scene_data.right_image
        ground_truth_disp_image = scene_data.disp_image
        ndisp = scene_data.ndisp # a conservative bound on the number of disparity levels
        
        # run window method with window size = 5, 15, 35
        for window_s in [5, 15, 35]:
            disparity_map = helper.window_disparity(left_image, right_image, window_size=window_s, min_disparity=0,
                                         max_disparity=ndisp-1, consensus_tol=5, normalize=False)
            
            
            # replace occluded pixels with 0
            disparity_map[disparity_map == 1<<30] = 0
            # normalize the disparity map to 8-bit
            disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # plot the disparity map using cv2
            cv2.imshow('Disparity Map', disparity_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # save the disparity map as png
            cv2.imwrite(f'disparity_map_window_{scene_name}_{window_s}.png', disparity_map)
            
            # present result from a different view and save as png result
            new_image = helper.present_view(left_image, right_image, disparity_map, viewpoint_shift)
            # write to png file
            cv2.imwrite(f'new_view_window_{scene_name}_{window_s}.png', new_image)
            
            # evaluate the disparity map and report the results write to file
            for method in eval_methods:
                error = helper.evaluation_disparity(ground_truth_disp_image, disparity_map, method=method)
                with open('window_method_evaluation_results.txt', 'a') as f:
                    f.write(f'{scene_name}_{window_s}: {method}: {error}\n')


            
    
def window_method_results():
    """
    print the results of the window method, save disparity maps as png, scale to 8-bit
    present result from a different view and save as png result
    report the evaluation results for all image pairs
    
    """
    
def graphcut_method_results():
    """
    print the results of the graphcut method, save disparity maps as png, scale to 8-bit
    present result from a different view and save as png result
    report the evaluation results for all image pairs
    
    """    
    
    
if __name__ == "__main__":
    # test()
    
    download_data() # Get list of scenes in Milddlebury's stereo training dataset save to local for next steps
    window_method_demo() # vary window size and print results
    window_method_results() #
    graphcut_method_results() #