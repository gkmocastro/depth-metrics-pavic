import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob


    
    
def calculate_delta(delta2, delta1):
    a, b = affine_least_squares(delta1, delta2)
    pred = delta1 * a + b
    max_ratio = np.maximum(delta2 / pred, pred / delta2)
    delta = max_ratio < 1.25
    num_true_values = np.count_nonzero(delta)
    measures = num_true_values / delta1.size

    return measures


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
    print(f"Unique values: {depth_dict['unique_values']}")
    print(f"how many uniques: {depth_dict['num_uniques']}")
    print(f"Max: {depth_dict['max']}")
    print(f"Min: {depth_dict['min']}")
    print(f"shape: {depth_dict['shape']}")
    print(f"Has nan: {depth_dict['has_nan']}")
    print(f"Dtype: {depth_dict['dtype']}")

    return depth_dict

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



