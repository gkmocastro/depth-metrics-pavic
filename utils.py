import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
import matplotlib.ticker as ticker


def depth_infos(depth):

    depth_dict = {
        "unique_values": np.unique(depth),
        "num_uniques": len(np.unique(depth)),
        "max": np.nanmax(depth),
        "min": np.nanmin(depth),
        "shape": depth.shape,
        "has_nan": np.isnan(depth).any(),
        "dtype": depth.dtype
    }


    print('---- Depth Report ----')
    print()
    #print(f"Unique values: {depth_dict['unique_values']}")
    print(f"how many uniques: {depth_dict['num_uniques']}")
    print(f"Max: {depth_dict['max']}")
    print(f"Min: {depth_dict['min']}")
    print(f"shape: {depth_dict['shape']}")
    #print(f"Has nan: {depth_dict['has_nan']}")
    print(f"Dtype: {depth_dict['dtype']}")

    return None

def show_rgbd(depth, image, figsize=(16,8), fontsize=16):
    depth_infos(depth)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.imshow(image)
    ax1.set_title("RGB", fontsize=fontsize)
    ax2.imshow(depth, cmap="gray")
    ax2.set_title("Depth", fontsize=fontsize)

    plt.tight_layout()

    plt.show()


def get_sorted_files(directory, endswith):
    """
    Get a sorted list of filenames within a specified folder and subfolders with an determined ending

    Args:
        directory: folder to search
        extension: desired pattern to match

    Returns: A sorted list of filenames with the specified ending
    """
    pattern = os.path.join(directory, "**", f"*{endswith}")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)

def normal(arr):
    return np.uint8(((arr - arr.min())/(arr.max() - arr.min()))*255)

def show_pred_gt(rgb, groundtruth, pred, figsize=(24,8), fontsize=16):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.imshow(rgb)
    ax1.set_title("RGB", fontsize=fontsize)
    ax2.imshow(normal(groundtruth), cmap="gray")
    ax2.set_title("GT", fontsize=fontsize)
    ax3.imshow(normal(pred), cmap="gray")
    ax3.set_title("Pred", fontsize=fontsize)

    plt.tight_layout()

    plt.show()

    depth_infos(groundtruth)
    depth_infos(pred)



def abs_rel_error(img1, img2):
    """
    Calculates the absolute relative error between two images.

    Args:
    img1: The first image as a NumPy array.
    img2: The second image as a NumPy array.

    Returns:
    The absolute relative error between the two images.
    """

    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)

    diff = np.abs(img1 - img2)
    rel_diff = diff / (img2 + 1e-8)  # Add a small value to avoid division by zero
    abs_rel = np.mean(rel_diff)
    return abs_rel

def cap_values(image, lower_percentile=0, upper_percentile=99):
    """
    Caps the values of an image array based on specified lower and upper percentiles.

    This function calculates the values at the specified lower and upper percentiles 
    of the input image array and then caps the image values. Any pixel value below the 
    lower percentile is set to the lower percentile value, and any value above the 
    upper percentile is set to the upper percentile value.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image array to be capped.
    
    lower_percentile : float, optional
        The lower percentile value for capping. Any pixel value below this percentile 
        will be set to the corresponding percentile value. Default is 0.
    
    upper_percentile : float, optional
        The upper percentile value for capping. Any pixel value above this percentile 
        will be set to the corresponding percentile value. Default is 99.

    Returns:
    --------
    numpy.ndarray
        The capped image array with pixel values adjusted to be within the specified 
        percentile range.
    """
    # Calculate the lower and upper percentile values
    lower_value = np.percentile(image, lower_percentile)
    upper_value = np.percentile(image, upper_percentile)

    # Cap the values
    capped_image = np.clip(image, lower_value, upper_value)

    return capped_image

def plot_histogram(image, ):
    # Flatten the image to a 1D array
    flattened_image = image.flatten()
    bins = 1000
    # Calculate the minimum and maximum values of the image
    min_val = np.min(flattened_image)
    max_val = np.max(flattened_image)
    
    # Plot the histogram
    plt.hist(flattened_image, bins=bins, range=(min_val, max_val), color='blue', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    plt.show()

def normalize_depth(depth):
   maxd = depth.max()
   mind = depth.min()

   return (depth - mind)/(maxd - mind)


def depth_report(rgb, depth, pred, cap=False, uint=True):
    """
    Generates a visual and statistical report for RGB, depth, and predicted depth images.

    This function creates a 2x2 plot that includes the RGB image, the predicted depth image, 
    the actual depth image, and a histogram of the depth values. It also optionally normalizes 
    the depth and predicted depth images for better visualization. 

    Parameters:
    -----------
    rgb : numpy.ndarray
        The input RGB image array.
    
    depth : numpy.ndarray
        The ground truth depth image array.
    
    pred : numpy.ndarray
        The predicted depth image array.
    
    cap : bool, optional
        If True, caps the values of the depth image to a specified percentile range (1st to 99th 
        percentile) for visualization purposes. Default is False.
    
    uint : bool, optional
        If True, converts the depth and predicted depth images to `uint8` format for better 
        visualization. Default is True.

    Returns:
    --------
    None
        The function displays the report with the RGB image, predicted depth, depth histogram, 
        and the depth image visualization. It does not return any value.
    """
    flattened_image = depth.flatten()
    bins = 1000
    # Calculate the minimum and maximum values of the image
    min_val = np.min(flattened_image)
    max_val = np.max(flattened_image)
    

    
    if uint: # if uint=true, transforma tudo para uint8 para visualizar legal
        pred_vis = normal(pred)
        if cap:
            depth_vis = cap_values(depth, 1, 99)
            depth_vis = normal(depth_vis)
        else:
            depth_vis = normal(depth)
    else: 
        pred_vis = pred
        depth_vis = depth
    

    fig, axes = plt.subplots(2,2, figsize=(8,6))

    axes[0,0].imshow(rgb)
    axes[0,0].set_title("RGB")
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])

    axes[0,1].imshow(pred_vis, cmap='gray')
    axes[0,1].set_title("Pred (Visualization)")
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])

    axes[1,1].hist(flattened_image, bins=bins, range=(min_val, max_val), color='blue', alpha=.7)
    axes[1,1].set_title("Depth Histogram")
    axes[1,1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axes[1,1].yaxis.get_major_formatter().set_scientific(True)
    axes[1,1].yaxis.get_major_formatter().set_powerlimits((0, 0))
    

    axes[1,0].imshow(depth_vis, cmap="gray")
    axes[1,0].set_title("Depth (Visualization)")
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    depth_infos(depth)
    depth_infos(pred)


def generate_flat_array(depth, mask):
    flat_depth = depth.flatten()
    flat_mask = mask.astype(bool).flatten()
    return flat_depth[flat_mask]


def align_depth(gt, pred, mask, return_depth=False, mask_output=False):
    flat_gt_masked = generate_flat_array(gt, mask)
    flat_pred_masked = generate_flat_array(pred, mask)
    A = np.vstack([flat_gt_masked, np.ones(len(flat_gt_masked))]).T
    s, t = np.linalg.lstsq(A, flat_pred_masked, rcond=None)[0]

    if return_depth:
        aligned =  1/((pred - t) / s)
    else:
        aligned = (pred - t) / s

    if mask_output:
        return aligned*(mask.reshape((mask.shape[0], mask.shape[1])))