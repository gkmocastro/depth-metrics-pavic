import numpy as np
from scipy.optimize import least_squares
from datasets import load_dataset
import os

depth_path = "/home/julio/Desktop/depth_npy"
# Carregar o conjunto de dados
ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation")
gt = ds["depth_map"]
delta1 = np.asarray(gt)

# Função para carregar os dados do segundo delta
def load_delta2(depth_path, idx):
    return np.load(os.path.join(depth_path, f"{idx + 1}_pred.npy"))

# Função para o cálculo dos mínimos quadrados invertidos
def inverse_affine_least_squares(delta1, delta2):
    # Definir a função de erro para os mínimos quadrados invertidos
    def error_func(params, x, y):
        a, b = params
        x_pred = a * y + b
        return np.ravel(x - x_pred)  # Flatten the array to 1D

    # Parâmetros iniciais
    initial_params = [1.0, 0.0]  # Suposição inicial de coeficientes

    # Executar os mínimos quadrados invertidos
    result = least_squares(error_func, initial_params, args=(delta1, delta2))

    # Recuperar os coeficientes ajustados
    a, b = result.x

    return a, b

# Iterar sobre os dados e calcular os mínimos quadrados invertidos para cada par delta1 e delta2
for idx in range(len(ds)):
    delta2 = load_delta2(depth_path, idx)
    a, b = inverse_affine_least_squares(delta1, delta2)
    print(f"Iteração {idx}: Coeficiente a:", a)
    print(f"Iteração {idx}: Coeficiente b:", b)
