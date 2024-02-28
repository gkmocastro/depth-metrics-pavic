import numpy as np

def calculate_absrel(gt_data, pred_data):
    
    absrel = np.mean(np.abs(gt_data - pred_data) / gt_data)
    
    return absrel


def calculate_delta(gt_data, pred_data):
    # Calcular Delta
    max_ratio = np.maximum(gt_data / pred_data, pred_data / gt_data)
    delta = max_ratio < 1.25

    # Contar quantos valores resultaram em True
    num_true_values = np.count_nonzero(delta)
    measures = num_true_values/ gt_data.size

    return delta, measures





