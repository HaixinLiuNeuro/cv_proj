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
OUTPUT_FOLDER = os.path.join(os.getcwd(),"output") #Path to save output images
# make output folder if it does not exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


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

    
def window_method_results():
    """
    print the results of the window method, save disparity maps as png, scale to 8-bit
    present result from a different view and save as png result
    report the evaluation results for all image pairs
    
    """

    viewpoint_shift = 0.9
    eval_methods = ['avg_abs', 'rms', 'A50', 'A90', 'A95', 'A99', 'bad0.5', 'bad1.0', 'bad2.0']
    window_sizes = [5, 55, 95, 155, 255]
    # use a dictionary to store the results of errors:
    # keys are the evaluation methods and values are the errors
    # keys also include the scene name and window size
    error_results = {}
    for method in eval_methods:
        error_results[method] = {}
        for scene_info in Dataset.get_training_scene_list():
            scene_name = scene_info.scene_name
            dataset_type = scene_info.dataset_type
            error_results[method][scene_name] = {}
            for window_s in [5, 15, 35]:
                error_results[method][scene_name][window_s] = np.inf
    
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
        # commandline print to show progress
        print(f'Processing {scene_name}')
        
        for window_s in window_sizes:
            print(f'Processing {scene_name} with window size {window_s}')

            disparity_map = helper.window_disparity(left_image, right_image, window_size=window_s, min_disparity=0,
                                         max_disparity=ndisp, consensus_tol=5, normalize=False)
            # occluded pixels are 0
            
            # normalize the disparity map to 8-bit
            disparity_map_norm = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # plot the disparity map using cv2
            """
            cv2.imshow('Disparity Map', disparity_map_norm)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            # save the disparity map as png
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, 
                                     f'disparity_map_window_{scene_name}_{window_s}_norm.png'), 
                                     disparity_map_norm)
            # TODO to make a new experiment to just do a few images
            # # present result from a different view and save as png result
            # new_image = helper.present_view(left_image, right_image, disparity_map, viewpoint_shift)
            # # write to png file
            # cv2.imwrite(os.path.join(OUTPUT_FOLDER, 
            #                          f'new_view_window_{scene_name}_{window_s}.png'), new_image)
            """
            cv2.imshow('present view', new_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            # evaluate the disparity map and report the results write to file
            for method in eval_methods:
                error = helper.evaluation_disparity(ground_truth_disp_image, disparity_map, method=method)
                error_results[method][scene_name][window_s] = error
                with open(os.path.join(OUTPUT_FOLDER, 'window_method_evaluation_results.txt'), 'a') as f:
                    f.write(f'{scene_name}_{window_s}: {method}: {error}\n')

    # perform statistical analysis for the error results:
    # report the mean and standard deviation of the errors for each evaluation method
    for method in eval_methods:
        # calculate mean and standard deviation
        errors = [error_results[method][scene_name][window_s] for scene_name in Dataset.get_training_scene_list() for window_s in window_sizes]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        with open(os.path.join(OUTPUT_FOLDER, 'window_method_evaluation_stats.txt'), 'a') as f:
            f.write(f'{method}: mean error: {mean_error}, std error: {std_error}\n')
    
def graphcut_method_results():
    """
    print the results of the graphcut method, save disparity maps as png, scale to 8-bit
    present result from a different view and save as png result
    report the evaluation results for all image pairs
    
    """    
    
    
if __name__ == "__main__":
    # test()
    
    download_data() # Get list of scenes in Milddlebury's stereo training dataset save to local for next steps
    # window_method_demo() # vary window size and print results
    window_method_results() #
    # present_from_another_view_demo()
    # graphcut_method_results() #