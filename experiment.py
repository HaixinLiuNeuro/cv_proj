"""
This is the main experiment script

"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import helper_classes as helper

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
    

if __name__ == "__main__":
    test()
    # Get list of scenes in Milddlebury's stereo training dataset
    download_data()
    window_method_demo() # vary window size and print results
    graphcut_method_demo() # vary window size and print results
    
    # 