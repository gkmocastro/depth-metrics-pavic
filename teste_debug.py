import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from utils import (get_sorted_files,  
                   abs_rel_error, 
                   cap_values, 
                   plot_histogram,  
                   depth_report, 
                   align_depth,
                   calculate_delta)


diode_path = Path("/home/gustavo/workstation/depth_estimation/data/datasets_quali/DIODE/")
diode_indoor = diode_path / "val" / "indoors"
diode_outdoor = diode_path / "val" / "outdoor"
diode_preds = Path("/home/gustavo/workstation/depth_estimation/data/outputs/DIODE-Anythingv2/npy/")

filenames_img = get_sorted_files(diode_path, ".png")
filenames_depth = get_sorted_files(diode_path, "_depth.npy")
filenames_mask = get_sorted_files(diode_path, "_depth_mask.npy")
filenames_preds_indoor = get_sorted_files(diode_preds, ".npy")


index = 100
index = 600
index = 680
pred_diode = np.load(filenames_preds_indoor[index])
groundtruth = np.squeeze(np.load(filenames_depth[index]))
rgb = np.array(Image.open(filenames_img[index]))
H,W = groundtruth.shape
mask = np.load(filenames_mask[index]).reshape((H, W))


mask_pred = np.ones_like(pred_diode)
mask_pred[pred_diode == 0] = 0


groundtruth_masked = np.zeros_like(groundtruth)
groundtruth_masked[mask == 1] = groundtruth[mask == 1]

disparity_gt_masked = np.zeros_like(groundtruth)
disparity_gt_masked[mask == 1] = 1.0 / groundtruth[mask == 1]

pred_depth = np.zeros_like(groundtruth) 
pred_depth[mask_pred == 1] = 1.0 / pred_diode[mask_pred == 1]

groundtruth_capped_upper = cap_values(groundtruth, 0, 99)
disparity_capped_upper = cap_values(disparity_gt_masked, 0, 99)


depth_aligned_masked = align_depth(disparity_gt_masked, pred_diode, mask, return_depth=False, mask_output=True)

depth_report(rgb, disparity_capped_upper ,depth_aligned_masked, cap=False, uint=False)