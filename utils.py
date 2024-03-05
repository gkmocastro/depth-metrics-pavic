import numpy as np
from scipy.optimize import least_squares
from datasets import load_dataset
import os

def affine_least_squares(delta1, delta2):
    def error_func(params, x, y):
        a, b = params
        y_pred = a * x + b
        return np.ravel(y - y_pred)  

    initial_params = [1.0, 0.0]  
    result = least_squares(error_func, initial_params, args=(delta1, delta2))

    a, b = result.x
    return a, b

def calculate_absrel(delta2, delta1):
    a, b = affine_least_squares(delta1, delta2)
    pred = delta1 * a + b
    absrel = np.mean(np.abs(delta2 - pred) / delta2)

    return absrel
def calculate_delta(delta2, delta1):
    a, b = affine_least_squares(delta1, delta2)
    pred = delta1 * a + b
    max_ratio = np.maximum(delta2 / pred, pred / delta2)
    delta = max_ratio < 1.25
    num_true_values = np.count_nonzero(delta)
    measures = num_true_values / delta1.size

    return measures




