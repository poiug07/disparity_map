import numpy as np
import cv2

def inpaint(disparity_map):
    """
    Naive Stroke inpainting.
    """
    # Create a binary mask where 0 pixels indicate missing data
    mask = np.zeros_like(disparity_map)
    mask[disparity_map < 70] = 255

    # Apply Inpainting algorithm to fill in missing values
    inpainting_method = cv2.INPAINT_NS
    inpainting_radius = 5
    return cv2.inpaint(disparity_map, mask, inpainting_radius, inpainting_method)

def mean_filter(disparity_map):
    """
    Apply mean filter to dispary map.
    """
    return cv2.blur(disparity_map, (5, 5))

def inpaint_mean(disparity_map):
    """
    Sequentially apply Naive Stroke inpainting and then mean filtering to disparity map.
    """
    # Create a binary mask where 0 pixels indicate missing data
    mask = np.zeros_like(disparity_map)
    mask[disparity_map < 70] = 255

    # Apply Inpainting algorithm to fill in missing values
    inpainting_method = cv2.INPAINT_NS
    inpainting_radius = 5
    inpaint = cv2.inpaint(disparity_map, mask, inpainting_radius, inpainting_method)

    return cv2.blur(inpaint, (5, 5))

def fill_in_missing(source, dest):
    """
    Filles in missing values in source by values in dest.
    """
    mask = dest < 70
    dest[mask] = source[mask]
    return dest