import numpy as np
#no código do absrel tá faltando divir M que é o total de pixels
def calculate_absrel(gt_data, pred_data):
    # Calcular AbsRel
    absrel = np.mean(np.abs(gt_data - pred_data) / gt_data)
    
    return absrel

#no código do delta adicionar linhas para calcular a proporção por de false e verdadeiro por dado
def calculate_delta(gt_data, pred_data):
    # Calcular Delta
    max_ratio = np.maximum(gt_data / pred_data, pred_data / gt_data)
    delta = max_ratio < 1.25

    return delta





