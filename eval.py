import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import utils as U

diode_path = Path("/home/gustavo/workstation/depth_estimation/data/datasets_quali/DIODE/")
# diode_indoor = diode_path / "val" / "indoors"
# diode_outdoor = diode_path / "val" / "outdoor"
diode_preds = Path("/home/gustavo/workstation/depth_estimation/data/outputs/DIODE-Anythingv2/npy/")



def eval_diode(gt_path, mask_path, pred_path):

    filenames_depth = U.get_sorted_files(gt_path, "_depth.npy")
    filenames_mask = U.get_sorted_files(mask_path, "_depth_mask.npy")
    filenames_preds = U.get_sorted_files(pred_path, ".npy")

    abs_rel = 0
    delta = 0
    num_imgs = len(filenames_depth)
    for index in range(num_imgs):
        mask_pred = np.ones_like(pred_diode)
        mask_pred[pred_diode < 1e-1] = 0
        mask_full = mask * mask_pred

        gt_depth_masked = np.zeros_like(groundtruth)
        gt_depth_masked[mask_full == 1] = groundtruth[mask_full == 1]

        gt_disp_masked = np.zeros_like(groundtruth)
        gt_disp_masked[mask_full == 1] = 1.0 / groundtruth[mask_full == 1]

        # pred_depth = np.zeros_like(groundtruth) 
        # pred_depth[mask_full == 1] = 1.0 / pred_diode[mask_full == 1]

        pred_diode = np.load(filenames_preds[index])
        groundtruth = np.squeeze(np.load(filenames_depth[index]))
        H,W = groundtruth.shape
        mask = np.load(filenames_mask[index]).reshape((H, W))

        x_0, x_1 = U.compute_scale_and_shift(pred_diode, gt_disp_masked, mask_full)

        prediction_aligned = x_0 * pred_diode + x_1
        prediction_aligned = np.squeeze(prediction_aligned)*mask_full

        depth_aligned_masked = np.zeros_like(pred_diode)
        depth_aligned_masked[mask_full == 1] = 1 / prediction_aligned[mask_full == 1]
    
        abs_rel += U.abs_rel_error_mask(depth_aligned_masked, gt_depth_masked, mask_full)
        delta += U.calculate_delta(depth_aligned_masked, gt_depth_masked, mask_full, threshold=1.25)

    return abs_rel/num_imgs, delta/num_imgs

if __name__ == '__main__':
    eval_diode(diode_path, diode_path, diode_preds)