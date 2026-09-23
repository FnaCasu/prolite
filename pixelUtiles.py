import numpy as np

def circular_mask(data, center, radius):
    h, w = data.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def inverse_circular_mask(data, center, radius):
    h, w = data.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center >= radius
    return mask

def create_inverse_annular_mask(data, r_int, r_ext, center=None):
    h, w = data.shape
    Y, X = np.ogrid[:h, :w]
    if center is None:
        center = [h/2, w/2]
    else:
        center = np.array(center)
        if center.shape[0]==1:
            center = np.array([center, center])
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask_ext = dist_from_center <= r_ext
    mask_int = dist_from_center >= r_int
    mask = np.full_like(mask_ext, True, dtype='bool')
    mask[mask_ext*mask_int] = False
    return mask

def find_bound_regions(xx, yy, mayor_dimension, minor_dimension):# Compute ideal range for data-cutting
    # Standard range in the mayor region
    XX_min = xx - int(minor_dimension/2)
    XX_max = xx + int(minor_dimension/2) + minor_dimension%2
    YY_min = yy - int(minor_dimension/2)
    YY_max = yy + int(minor_dimension/2) + minor_dimension%2
    # Standard range in the minor ragion
    x_min = 0
    x_max = minor_dimension
    y_min = 0
    y_max = minor_dimension
    # Check if the cut it is outside the data, in case shift the output region to keep the centering
    if XX_min < 0:
        x_min = - XX_min
        XX_min = 0
    if YY_min < 0:
        y_min = - YY_min
        YY_min = 0
    if XX_max > mayor_dimension :
        x_max = mayor_dimension - XX_max
        XX_max = mayor_dimension
    if YY_max > mayor_dimension:
        y_max = mayor_dimension - YY_max
        YY_max = mayor_dimension
    # Pack elements
    mayor_limits = np.zeros((2,2), dtype=np.int16)
    mayor_limits[0] = [XX_min, XX_max]
    mayor_limits[1] = [YY_min, YY_max]
    minor_limits = np.zeros((2,2), dtype=np.int16)
    minor_limits[0] = [x_min, x_max]
    minor_limits[1] = [y_min, y_max]
    # return elements
    return mayor_limits, minor_limits