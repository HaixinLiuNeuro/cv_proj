""" helper classes

"""

import os 
import numpy as np
import cv2
from stereomideval.dataset import Dataset # for getting data
# for graph cut
import maxflow 
import multiprocessing as mp # parallel processing
import heapq # for heap queue



def download_dataset(dataset_folder):
    """ download dataset from middlebury servers

    Args:
        scene_name (str): scene name
        dataset_folder (str): dataset folder
        dataset_type (str): dataset type

    Returns:
        None
    """
    
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Get list of scenes in Milddlebury's stereo training dataset and iterate through them
    for scene_info in Dataset.get_training_scene_list():
        scene_name=scene_info.scene_name
        dataset_type=scene_info.dataset_type
        # Download dataset from middlebury servers
        # will only download it if it hasn't already been downloaded
        print("Downloading data for scene '"+scene_name+"'...")
        Dataset.download_scene_data(scene_name,dataset_folder,dataset_type)



def SSD_disparity_map(left_image_ori, right_image_ori, window_size=5, 
                      min_disparity=0, max_disparity=50, method='ssd_fast'):
    """ SSD disparity map computation

    Args:
        left_image (np.array): left image grayscale
        right_image (np.array): right image
        window_size (int): window size
        min_disparity (int): minimum disparity
        max_disparity (int): maximum disparity
        method (str): method name, 'ssd_slow', 'ssd_fast', 'ssd_fast_reverse'
                
    Returns:
        
    """
    # check window size must be odd
    if window_size % 2 == 0:
        raise ValueError('Window size must be odd')
    
    # check image size
    if left_image_ori.shape != right_image_ori.shape:
        raise ValueError('Left and right images must have the same shape')
    
    # check min_disparity and max_disparity
    if min_disparity < 0:
        raise ValueError('Minimum disparity must be greater or equal to zero')
    if max_disparity < 0:
        raise ValueError('Maximum disparity must be greater or equal to zero')
    if max_disparity < min_disparity:
        raise ValueError('Maximum disparity must be greater or equal to minimum disparity')
    
    assert left_image_ori.shape[1] >= window_size, "window too big"
    
    assert method in ['ssd_slow', 'ssd_fast', 'ssd_fast_reverse'], "method not supported"
        
    # get image dimensions
    height = left_image_ori.shape[0]
    width = left_image_ori.shape[1]
    
    # initialize disparity map
    disparity_map = np.zeros((height, width), dtype=np.float64) 
    left_image = left_image_ori.copy().astype(np.float64)
    right_image = right_image_ori.copy().astype(np.float64)
    
    # choose method pass the function handle
    if method == 'ssd_slow':
        method = ssd
        # half window
        half_window = window_size // 2
        # iterate over height: slow method
        for y in range(half_window, height - half_window):
            for x in range(half_window, width - half_window):
                # define search range
                search_range_max = min(max_disparity, width - x)
                search_range_min = min(min_disparity, width - x)
                
                # initialize best match and best cost
                best_match = 0
                best_cost = float('inf')
                
                # iterate over search range
                for d in range(search_range_min, search_range_max):
                    if x - half_window - d < 0:
                        continue
                    
                    # get left and right patches
                    left_patch = left_image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
                    right_patch = right_image[y-half_window:y+half_window+1, x-half_window-d:x+half_window+1-d]
                    
                    # compute cost
                    cost = method(left_patch, right_patch)
                    
                    # update best match and best cost
                    if cost < best_cost:
                        best_cost = cost
                        best_match = d
                
                # update disparity map
                disparity_map[y, x] = best_match
    elif method == 'ssd_fast':
        # bound max_disparity by width
        max_disparity = min(max_disparity, width)    
        # vectorized method using cv2.filter2D
        # define template window
        template_window = np.ones((window_size, window_size), dtype=np.float64)
        # cache ssd map for each d        
        best_ssd_map = np.ones((height, width), dtype=np.float64) * np.inf
        
        for d in range(min_disparity, max_disparity):
            # Shift right image to the right by 'd' pixels
            shifted_right_image = np.roll(right_image, d, axis=1)
            shifted_right_image[:, 0:d+1] = 0  # Padding zero on the left edge
            
            # Compute squared differences
            squared_diff = (left_image - shifted_right_image) ** 2
            
            # convolve using cv2.filter2D
            ssd_map = cv2.filter2D(squared_diff, -1, template_window, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)
            # if 3rd dimension is > 1, then we need to sum the 3rd dimension
            if len(ssd_map.shape) > 2:
                ssd_map = np.sum(ssd_map, axis=2)                
            
            mask = ssd_map < best_ssd_map
            
            # get minimum ssd map
            best_ssd_map[mask] = ssd_map[mask]
            disparity_map[mask] = d * mask[mask]
            
    elif method == 'ssd_fast_reverse': # compute disparity from right to left
        max_disparity = min(max_disparity, width)    
        # vectorized method using cv2.filter2D
        # define template window
        template_window = np.ones((window_size, window_size), dtype=np.float64)
        # cache ssd map for each d        
        best_ssd_map = np.ones((height, width), dtype=np.float64) * np.inf
        
        for d in range(min_disparity, max_disparity):
            # Shift left image to the left by 'd' pixels
            shifted_left_image = np.roll(left_image, -d, axis=1)
            shifted_left_image[:, (width-d):] = 0  # Padding zero on the right edge
            
            squared_diff = (shifted_left_image - right_image) ** 2
            
            ssd_map = cv2.filter2D(squared_diff, -1, template_window, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)
            
            if len(ssd_map.shape) > 2:
                ssd_map = np.sum(ssd_map, axis=2) 
                
            mask = ssd_map < best_ssd_map
            best_ssd_map[mask] = ssd_map[mask]
            disparity_map[mask] = d * mask[mask]
            
    else:
        raise ValueError('Unknown method')
    
    # convert disparity map to uint8
    # disparity_map = (disparity_map / max_disparity) * 255
    # disparity_map = disparity_map.astype(np.uint8)
                
    return disparity_map

def ssd(left_patch, right_patch):
    """ compute sum of squared differences for demostration purpose with left and right patches input

    Args:
        left_patch (np.array): left patch
        right_patch (np.array): right patch

    Returns:
        float: sum of squared differences
    """
    # check if patches have the same shape
    if left_patch.shape != right_patch.shape:
        print(left_patch.shape, right_patch.shape)        
        raise ValueError('Patches must have the same shape')
    
    return np.sum(np.square(left_patch - right_patch))
        


def window_disparity(left_image_ori, right_image_ori, window_size=5, 
                      min_disparity=0, max_disparity=50, consensus_tol=5, normalize=False):
    """ use get_disparity_map to compute disparity map with concensus matching
        if both side disparity is the within a tolerance, then the disparity is valid, use the mean of the two disparity
        else it is occluded, set to 0        
        
    Args:
        left_image (np.array): left image grayscale
        right_image (np.array): right image
        window_size (int): window size
        min_disparity (int): minimum disparity
        max_disparity (int): maximum disparity
        consensus_tol (int): consensus tolerance
        normalize (bool): normalize disparity map to 0-255
    Returns:
        disparity_map (np.array): disparity map            
    """
    disparity_L2R = SSD_disparity_map(left_image_ori, right_image_ori, window_size, 
                      min_disparity, max_disparity, method='ssd_fast')
    disparity_R2L = SSD_disparity_map(left_image_ori, right_image_ori, window_size, 
                      min_disparity, max_disparity, method='ssd_fast_reverse')
    
    consensus_map = np.zeros_like(disparity_L2R)
    consensus_map = np.abs(disparity_L2R - disparity_R2L) <= consensus_tol
    disparity_map = np.zeros_like(disparity_L2R)
    disparity_map[consensus_map] = (disparity_L2R[consensus_map] + disparity_R2L[consensus_map]) // 2
    disparity_map[~consensus_map] = 0
    
    if normalize:
        disparity_map = (disparity_map / np.max(disparity_map)) * 255
        disparity_map = disparity_map.astype(np.uint8)
        
    return  disparity_map

def evaluation_disparity(ground_truth, disparity, method='rms'):
    """ Evaluate the disparity map against the ground_truth image
        10 disparity accuracy measures, including 
        bad pixel percentages for thresholds 0.5, 1.0, 2.0, and 4.0; 
        average absolute and RMS disparity errors; 
        and error quantiles A50 (median error), A90, A95, and A99.
    
    Arg:
        ground_truth (np.array): ground truth disparity map
        disparity (np.array): disparity map
        method (str): method name
            
    Return:
        float: disparity accuracy measure
        
    """
    
    assert ground_truth.shape == disparity.shape, 'Ground truth and disparity map must have the same shape'
    
    assert method in ['avg_abs', 'rms', 'A50', 'A90', 'A95', 'A99', 'bad0.5', 'bad1.0', 'bad2.0'], "method not supported"
    
    if 'A' in method: # quantile method if contains A
        quantile = int(method[1:])
        return np.percentile(np.abs(ground_truth - disparity), quantile)
    elif 'bad' in method: # bad pixel method
        threshold = float(method[3:])
        bad_pixels = np.abs(ground_truth - disparity) > threshold
        return np.sum(bad_pixels) / bad_pixels.size
    elif method == 'avg_abs':
        return np.mean(np.abs(calc_diff(ground_truth, disparity)))
    elif method == 'rms':
        return np.sqrt(np.mean(np.square(calc_diff(ground_truth, disparity))))
    else:
        raise ValueError('Unknown method')
    
def calc_diff(ground_truth, disparity):
    """ Calculate the difference between ground truth and disparity map
    
    Args:
        ground_truth (np.array): ground truth disparity map
        disparity (np.array): disparity map
        
    Returns:
        np.array: difference map
    """
    # check if ground truth and disparity map have the same shape
    assert ground_truth.shape == disparity.shape, 'Ground truth and disparity map must have the same shape'
    
    # replace NaN and inf with zeros
    ground_truth_copy = np.nan_to_num(ground_truth, nan=0.0, posinf=0.0, neginf=0.0)    
    disparity_copy = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.subtract(ground_truth_copy - disparity_copy)
    

def present_view(left_image, right_image, disparity_map, viewpoint_shift):
    """ present the image from a different view
    
    Args:
        viewpoint_shift: float, the factor to adjust the viewpoint; 
                             values <1.0 shift towards right, >1.0 shift towards left.
    Returns:
        new_image: np.array: new image view                             
    """
    # Initialize the new image with zeros (black)
    height, width = left_image.shape[:2]
    new_image = np.zeros_like(left_image)
    
    # Calculate the new x-coordinates for each pixel
    for y in range(height):
        for x in range(width):
            disparity = disparity_map[y, x]
            new_x = int(x + disparity * viewpoint_shift)

            # Check if the new x-coordinate is within the image bounds
            if 0 <= new_x < width:
                new_image[y, new_x] = left_image[y, x]
                
    return new_image
    
# class for graph cut
# Global constants
VAR_ALPHA = -1
VAR_ABSENT = -2
CUTOFF = 30
OCCLUDED = 1<<30

class GCKZ: 
    """
    GCGZ class is the implementation of Kolmogorov et al. 2014 Image Processing On Line paper:
    http://www.ipol.im/pub/art/2014/97/?utm_source=doi
    
    """
    
    def __init__(self, left_image, right_image):
        self.left_image = left_image.copy().astype(np.float64)
        self.right_image = right_image.copy().astype(np.float64)
        self.occluded = OCCLUDED # 1<<30
        self.maxIterations = 4
        self.alphaRange = 16
        # self.left_image_min, self.left_image_max = self.subPixel(self.left_image)
        # self.right_image_min, self.right_image_max = self.subPixel(self.right_image)
        # get I max and I min defined in the paper (BT dissimilarity measure)
        self.left_image_min, self.left_image_max = self.i_maxmin(self.left_image)
        self.right_image_min, self.right_image_max = self.i_maxmin(self.right_image)
        
        # auto calculate K, lambda1, lambda2, denominator as per the paper
        self.K, self.lambda1, self.lambda2, self.denominator = self.setupParameters()
        
        # initialize energy and disparity map
        self.energy = 0
        self.disparity_map = np.full((self.left_image.shape), self.occluded, dtype=np.int64)
        
        # to track modes of associated pixels
        self.vars0 = np.zeros((self.left_image.shape), dtype=np.int64)
        self.varsA = np.zeros((self.left_image.shape), dtype=np.int64)
        
        self.activePenalty = 0
        
        self.edgeThresh = 8 # as per the paper for smoothness term
        
    # function to calculate distance using  Birchfield & Tomasi's dissimilarities with KZ algo's addition
    def disBT(self, p, q):
        """ calculate Birchfield & Tomasi's dissimilarities with KZ algo's addition
        
        Args:
            p (tuple): pixel p in the left image
            q (tuple): pixel q in the right image
        Returns:
            disBT_l2r (float): dissimilarity from left to right
            disBT_r2l (float): dissimilarity from right to left
        """
        I1_p = self.left_image[p]
        I2_q = self.right_image[q]
        
        I2_max_q = self.right_image_max[q]
        I2_min_q = self.right_image_min[q]
        
        I1_max_p = self.left_image_max[p]
        I1_min_p = self.left_image_min[p]
        # construct the dissimilarity measure for left 2 right
        dis_measure = np.array([0, I1_p - I2_max_q, I2_min_q - I1_p])
        disBT_l2r = np.max(dis_measure)
        
        # right 2 left        
        dis_measure = np.array([0, I2_q - I1_max_p, I1_min_p - I2_q])
        disBT_r2l = np.max(dis_measure)
        
        
        return disBT_l2r, disBT_r2l
    
    def i_maxmin(self, im):
        """ get I max and min on the one-pixel-large-neighborhood
        """
        left = np.copy(im).astype(float)
        left[:,1:] += im[:,:-1]
        left[:,1:] /= 2
        # ? how about edge
        
        right = np.copy(im).astype(float)
        right[:,:-1] += im[:,1:]
        right[:,:-1] /=2
        
        up = np.copy(im).astype(float)
        up[1:,:] += im[:-1,:]
        up[1:,:] /=2
        
        down = np.copy(im).astype(float)
        down[:-1,:] += im[1:,:]
        down[:-1,:] /=2
        
        im_max = np.max(np.array([im, left, right, up, down]), axis=0)
        im_min = np.min(np.array([im, left, right, up, down]), axis=0)
        
        return im_max, im_min
    
    def trim_disBT(self, disBT_l2r, disBT_r2l):
        """ trimmed and symmetric version of BT dissimilarity
        
        """        
        
        disBT_trim = np.min(np.array([CUTOFF, disBT_l2r, disBT_r2l]))
        
        return disBT_trim        
            
    def setupParameters(self):
        """
        compute K, lambda1, lambda2, denominator as per the paper
        
        """
        
        k = max(3, (self.alphaRange+3) // 4)
        sum_kth_smallest = 0
        sum_num = self.left_image.shape[0] * (self.left_image.shape[1] - self.alphaRange)
        
        # for each pixel p (not close to border)
        # compute C(p) as the Kth smallest value of all the D(a) values for assignment a involving the pixel p
        for row, col in np.ndindex(self.left_image.shape[0], self.left_image.shape[1] - self.alphaRange):
            tmp_distance = np.empty(self.alphaRange+1)
            for d in range(0, self.alphaRange+1):
                disBT_l2r, disBT_r2l = self.disBT((row, col), (row, col + d))
                tmp_distance[d] = self.trim_disBT(disBT_l2r, disBT_r2l)
            kth_smallest = np.partition(tmp_distance, k-1)[k-1]
            sum_kth_smallest += kth_smallest
        # set K to the average of all C(p)    
        K = sum_kth_smallest/sum_num
        
        lambda2 = K / 5
        lambda1 = 3 * lambda2
        
        # as equation 36 in the paper
        N = []
        for i in range(1,17):
            first_numerator = round(i * K)
            first_term = abs(first_numerator / (i * K) - 1)
            second_numerator = round(i * lambda1)
            second_term = abs(second_numerator / (i * lambda1) - 1)
            third_numerator = round(i * lambda2)
            third_term = abs(third_numerator / (i * lambda2) - 1)
            # also cashe the values to fetch after find the min
            N.append((first_term+second_term+third_term, first_numerator, second_numerator, third_numerator, i))
        # sort by N and get the smallest
        params = sorted(N)[0]
        _, K, lamda1, lamda2, denominator = params
        
        return K, lamda1, lamda2, denominator
    
    def update_disparity_map(self, g, d):
        """
        use the disparity label (d) to update the disparity map attribute
        
        """
        
        vectorized_get_segment = np.vectorize(g.get_segment)
        
        # vars0 
        if self.vars0[self.vars0 >= 0].size > 0:
            mask = np.zeros(self.vars0.shape, dtype=bool)
            mask[self.vars0 >= 0] = vectorized_get_segment(self.vars0[self.vars0 >= 0])
            self.disparity_map[mask] = OCCLUDED
            
        # varsA 
        if self.varsA[self.varsA >= 0].size > 0:
            mask = np.zeros(self.varsA.shape, dtype=bool)
            mask[self.varsA >= 0] = vectorized_get_segment(self.varsA[self.varsA >= 0])
            self.disparity_map[mask] = d
        
    def add_data_occu_terms(self, g, d):
        """
        add occlusion terms to the graph
        
        """ 
        for p, disp in np.ndenumerate(self.disparity_map):
            # disp = self.disparity_map[p]
            
            if disp == d: # acctive assignment
                self.vars0[p] = VAR_ALPHA
                self.varsA[p] = VAR_ALPHA                
                # get distance and penalty
                q = (p[0], p[1] + disp) # y, x + disp
                disBT_l2r, disBT_r2l = self.disBT(p, q)
                disBT_trim = self.trim_disBT(disBT_l2r, disBT_r2l)
                penalty = self.denominator * disBT_trim - self.K
                self.activePenalty += penalty # accumulate active penalty
                # no new node
            else: # inactive
                # vars0
                if disp != self.occluded: # not occluded, add node and edge to sink
                    q = (p[0], p[1] + disp) # y, x + disp
                    disBT_l2r, disBT_r2l = self.disBT(p, q)
                    disBT_trim = self.trim_disBT(disBT_l2r, disBT_r2l)
                    penalty = self.denominator * disBT_trim - self.K
                    node_id = g.add_nodes(1)[0]
                    g.add_tedge(node_id, 0, penalty)
                    self.vars0[p] = node_id # cache node id to pixel position
                else: # occluded
                    self.vars0[p] = VAR_ABSENT
                
                # varsA new assignment
                q = (p[0], p[1] + d)
                # check if q is within the image
                if q[1] < self.left_image.shape[1]: # within the image
                    disBT_l2r, disBT_r2l = self.disBT(p, q)
                    disBT_l2r = self.trim_disBT(disBT_l2r, disBT_r2l)
                    penalty = self.denominator * disBT_l2r - self.K
                    node_id = g.add_nodes(1)[0]
                    g.add_tedge(node_id, penalty, 0)
                    self.varsA[p] = node_id                    
                else: # outside the image
                    self.varsA[p] = VAR_ABSENT        
        
    def add_smooth_terms(self, g, d):
        """
        add smoothness terms to the graph
        if p1 and p2 have the same disparity Esmooth = 0
        
        
        """
        for p1, _ in np.ndenumerate(self.disparity_map):
            # disp = self.disparity_map[p]            
        
            for i in range(2):
                # q2 = (q[0] + i, q[1] + i - 1)
                p2 = (p1[0] + i, p1[1] + i - 1)
                
                if p2[1] < self.left_image.shape[1] and p2[0] < self.left_image.shape[0] and \
                    p2[0] >= 0 and p2[1] >= 0: # valid p2
                
                    d_f_p1 = self.disparity_map[p1]
                    d_f_p2 = self.disparity_map[p2]
                    
                    alpha = d
                    
                    varsA_a1 = self.varsA[p1]
                    varsA_a2 = self.varsA[p2]
                    
                    vars0_a1 = self.vars0[p1]
                    vars0_a2 = self.vars0[p2]
                    
                    
                    
                    
                    # if disp1 = disp2, 
                    #   if both active then, Esmooth = 0  (self.vars0[p] >= 0 and self.vars0[p2] >= 0)
                    #   if both inactive then, Esmooth = 0 (self.vars0[p] < 0 and self.vars0[p2] < 0)
                    
                    # 1: a1, a2 A_alpha\f
                    # a1 = (p1, p1 + alpha) and a2 = (p2, p2 + alpha), 
                    # a1 is variable, a2 is variable
                    # varsA(p1) = a1, varsA(p2) = a2
                    # add pairwise assignment a1, a2, 0, weight, weight, 0
                    if varsA_a1 >= 0 and varsA_a2 >= 0 and \
                        p2[1]+alpha < self.left_image.shape[1] and p1[1]+alpha < self.left_image.shape[1]:
                    # varsA_a1 != VAR_ALPHA and varsA_a1 != VAR_ABSENT and varsA_a2 != VAR_ALPHA  and vars0_a2 != VAR_ABSENT:
                        weight = self.dist_dispairty(p1, p2, (p1[0], p1[1] + alpha), (p2[0], p2[1] + alpha))
                        self.add_edge_pairwise(g, varsA_a1, varsA_a2, 0, weight, weight, 0)
                    
                    # 2: a1 A_alpha\f, a2 A_alpha A(f)
                    # a1 = (p1, p1 + alpha) and a2 = (p2, p2 + alpha),
                    # a1 is variable, a2 remains active                                
                    # varsA(p1) = a1, varsA(p2) = VAR_ALPHA
                    # add tedge a1, weight, 0
                    if varsA_a1 >= 0 and varsA_a2 == VAR_ALPHA:
                        weight = self.dist_dispairty(p1, p2, (p1[0], p1[1] + alpha), (p2[0], p2[1] + alpha))
                        # add terminal edge
                        g.add_tedge(varsA_a1, weight, 0)
                            
                    
                    # 3: a1 A_O, a2 A_O
                    # a1 = (p1, p1 + d) and a2 = (p2, p2 + d),  d = d_f(p1) = d_f(p2)
                    # a1 is variable and a2 is variable
                    # vars0(p1) = o1, vars0(p2) = o2
                    # add pairwise assignment o1, o2, 0, weight, weight, 0
                    if d_f_p1 == d_f_p2 and vars0_a1 >= 0 and vars0_a2 >= 0 and \
                        p2[1] + d < self.left_image.shape[1] and p1[1] + d < self.left_image.shape[1]:
                        weight = self.dist_dispairty(p1, p2, (p1[0], p1[1] + d), (p2[0], p2[1] + d))
                        self.add_edge_pairwise(g, vars0_a1, vars0_a2, 0, weight, weight, 0)
                    
                    
                    # 4: a1 A_O, a2 A \ A_O
                    # a1 = (p1, p1 + d) and a2 = (p2, p2 + d),  d = d_f(p1) != d_f(p2)
                    # a1 is variable and a2 remains inactive
                    # vars0(p1) = o1, d_left(p1) != d_left(p2), p2+d_left(p1) belongs to the rigth image
                    # add tedge o1, weight, 0
                    if d_f_p1 != d_f_p2 and vars0_a1 >= 0 and vars0_a2 == VAR_ABSENT and \
                        p2[1] + d < self.left_image.shape[1] and p1[1] + d < self.left_image.shape[1]:
                        weight = self.dist_dispairty(p1, p2, (p1[0], p1[1] + d), (p2[0], p2[1] + d))
                        g.add_tedge(vars0_a1, weight, 0)
                
    def dist_dispairty(self, p1, p2, q1, q2):
        """
        Calculate smoothness term's weight
        Args:
        
        Returns:
            weight (float): smoothness term's weight
        """
        
        intensity_absdiff_I1 = np.abs(self.left_image[p1] - self.left_image[p2])
        intensity_absdiff_I2 = np.abs(self.right_image[q1] - self.right_image[q2])
        
        max_term = max(intensity_absdiff_I1, intensity_absdiff_I2)
        if max_term < self.edgeThresh:
            return self.lambda1 
        else:
            return self.lambda2
                
    def add_unique_terms(self, g, d):
        """
        add unique terms to the graph
        
        """
        for p, disp in np.ndenumerate(self.disparity_map):
            # disp = self.disparity_map[p]
            q = (p[0], p[1] + disp)
            
            if self.vars0[p] >= 0: # not absent
                var_A = self.varsA[p] # retrive the node id
                if var_A != VAR_ABSENT:
                    # add forbidden edge
                    self.add_edge_forbid01(g, self.vars0[p], var_A, OCCLUDED)
                p2 = (p[0], p[1] + disp - d)
                if p2[1] < self.left_image.shape[1] and p2[1] >= 0:
                    var_A = self.varsA[p2]
                    # add forbidden edge
                    self.add_edge_forbid01(g, self.vars0[p], var_A, OCCLUDED)
                    
    def add_edge_forbid01(self, g, n1, n2, OCCLUDED):
        """
        add forbidden edge between two nodes
        
        """
        g.add_edge(n1, n2, OCCLUDED, 0)
        
    def add_edge_pairwise(self, g, n1, n2, E00, E01, E10, E11):
        """
        add pairwise edge between two nodes
        
        """
        g.add_edge(n1, n2, 0, (E01+E10)-(E00+E11))
        g.add_tedge(n1, E11, E01)
        g.add_tedge(n2, 0, E00 - E01)
                        
        
    def compute_disparity_map(self):
        """
        main function to compute disparity map (loops)
        
        
        Returns:
            disparity_map (np.array): disparity map
        """
        max_disparity = self.alphaRange + 1 # including 0
        
        # reinitialize disparity map and energy
        self.disparity_map = np.full((self.left_image.shape), self.occluded, dtype=np.int64)
        self.energy = 0
        
        # track assignment
        done = np.full(max_disparity, False)
        
        # loop
        for i in range(self.maxIterations):
            # loop over all labels
            for d in range(max_disparity): # np.random.permutation(max_disparity):?
                if not done[d]:
                    self.activePenalty = 0 # initialize active penalty
                    # construct a graph for each
                    g = maxflow.Graph[int](2 * self.left_image.size, 12 * self.left_image.size) 
                    # each pixel has two nodes and 12 edges
                    
                    self.add_data_occu_terms(g, d)
                    self.add_smooth_terms(g, d)
                    self.add_unique_terms(g, d)
                    
                    new_energy = g.maxflow() + self.activePenalty
                    
                    if new_energy < self.energy:
                        self.energy = new_energy
                        # update disparity map
                        self.update_disparity_map(g, d)
                        done[:] = False # reset
                    done[d] = True
                    
                    # exit
                    if np.all(done):
                        return self.disparity_map
        # finished loops        
        return self.disparity_map
    
# helper functions to split images and compose result/final disparity map for GCKZ
def run_gckz(left_image, right_image):
    g = GCKZ(left_image, right_image)
    disparity_map = g.compute_disparity_map()
    
    return disparity_map
    
def process_image(left_image, right_image, n_cpu=-1):
    """ process image
    
    """
    
    if n_cpu <=0:
        n_cpu = max(1, mp.cpu_count()-1)
        
    left_strips = np.array_split(left_image, n_cpu)
    right_strips = np.array_split(right_image, n_cpu)
    
    with mp.Pool(processes=n_cpu) as pool:
        disparity_strips = pool.starmap(run_gckz, zip(left_strips, right_strips))
        
    disparity_map = np.vstack(disparity_strips)
    
    # turn occluded pixels to 0
    disparity_map[disparity_map == OCCLUDED] = 0
    
    return disparity_map
                    
                
        
    
# Cite:
# [BOYKOV04] Boykov, Y., Veksler, O., & Zabih, R. (2001). Fast approximate energy minimization via graph cuts. IEEE Transactions on Pattern Analysis and Machine Intelligence, 23(11), 1222-1239.
# helpful tips from peers:
# https://edstem.org/us/courses/51358/discussion/4616294?comment=11131939
