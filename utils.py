import numpy as np
import os
import matplotlib.pyplot as plt
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
    #print(f"shape: {depth_dict['shape']}")
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


# future work: use pathlib instead
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

def abs_rel_error_mask(img1, img2, mask):
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)

    rel_diff = np.zeros_like(img1)
    diff = np.zeros_like(img1)

    diff[mask == 1] = np.abs(img1[mask==1] - img2[mask==1])
    rel_diff[mask==1] = diff[mask==1] / (img2[mask==1])# + 1e-8)  # Add a small value to avoid division by zero
    return np.mean(rel_diff)

# def abs_rel_error(img1, img2):
#     img1 = np.squeeze(img1)
#     img2 = np.squeeze(img2)

#     diff = np.abs(img1 - img2)
#     rel_diff = diff / (img2 + 1e-8)  # Add a small value to avoid division by zero
#     abs_rel = np.mean(rel_diff)
#     return abs_rel

def cap_values(image, lower_percentile=0, upper_percentile=99):
    """
    Caps the values of an image array based on specified lower and upper percentiles.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image array to be capped.
    
    lower_percentile : float, optional
        The lower percentile value for capping. 
    
    upper_percentile : float, optional
        The upper percentile value for capping. 

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

def plot_histogram(image, mask):
    # Flatten the image to a 1D array
    flattened_image = generate_flat_array(image, mask)
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


def depth_report(rgb, depth, pred, mask, cap=False, uint=True, bins=1000):

    flat_depth = generate_flat_array(depth, mask)
    flat_pred = generate_flat_array(pred, mask)
    # Calculate the minimum and maximum values of the image
    min_depth = np.min(flat_depth)
    max_depth =  np.max(flat_depth)

    min_pred = np.min(flat_pred)
    max_pred =  np.max(flat_pred)

    
    flat_depth_cap = cap_values(flat_depth, 2, 98)
    flat_pred_cap = cap_values(flat_pred, 2, 98)

    #melhorar: nao funciona se quiser cap, mas nao uint, mas isso é irrelevante
    if uint: 
        if cap:
            pred_vis = cap_values(pred, 1, 99)
            pred_vis = normal(pred_vis)
            depth_vis = cap_values(depth, 1, 99)
            depth_vis = normal(depth_vis)
        else:
            depth_vis = normal(depth)
            pred_vis = normal(pred)
    else: 
        pred_vis = pred
        depth_vis = depth
    

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2,3, figsize=(12,8))

    ax0.imshow(rgb)
    ax0.set_title("RGB")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax3.imshow(mask, cmap="gray")
    ax3.set_title("Mask")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax2.imshow(pred_vis, cmap='gray')
    ax2.set_title("Pred (Visualization)")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax4.hist(flat_depth, bins=bins, range=(min_depth, max_depth), color='blue', alpha=.7)
    ax4.set_title("Depth Histogram")
    ax4.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax4.yaxis.get_major_formatter().set_scientific(True)
    ax4.yaxis.get_major_formatter().set_powerlimits((0, 0))


    ax5.hist(flat_pred, bins=bins, range=(min_pred, max_pred), color='blue', alpha=.7)
    ax5.set_title("Pred Histogram")
    ax5.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax5.yaxis.get_major_formatter().set_scientific(True)
    ax5.yaxis.get_major_formatter().set_powerlimits((0, 0))


    ax1.imshow(depth_vis, cmap="gray")
    ax1.set_title("Depth (Visualization)")
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    plt.show()
    depth_infos(flat_depth)
    depth_infos(flat_pred)


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
    

def calculate_delta(pred, gt, mask, threshold=1.25):

    err = np.zeros_like(pred, dtype=np.float64)

    err[mask == 1] = np.maximum(
        pred[mask==1] / gt[mask==1],
        gt[mask==1] / pred[mask==1],
    )

    err[mask == 1] = (err[mask == 1] < threshold)

    p = np.sum(err) / np.sum(mask)

    return 100 * np.mean(p)

    
def compute_scale_and_shift(prediction, target, mask):
    # h,w = prediction.shape
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, (0,1))
    a_01 = np.sum(mask * prediction, (0,1))
    a_11 = np.sum(mask, (0,1))

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, (0,1))
    b_1 = np.sum(mask * target, (0,1))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1