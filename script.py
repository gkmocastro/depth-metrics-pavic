import numpy as np
from datasets import load_dataset
import os
from utils import ( 
         calculate_absrel,
         calculate_delta,)


depth_path = "/home/julio/Desktop/depth_npy"
ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation")
all_absrel = []
all_delta = []
# Iterar sobre os dados e calcular as métricas para cada par de deltas
for idx, sample in enumerate(ds):
    # Carregar os deltas
    delta1 = np.load(os.path.join(depth_path, f"{idx + 1}_pred.npy"))
    delta2 = sample["depth_map"]
    
    # Calcular as métricas
    absrel = calculate_absrel(delta2, delta1)
    measures = calculate_delta(delta2, delta1)
    all_absrel.append(absrel)
    all_delta.append(measures)
    mean_absrel = np.mean(all_absrel)
    mean_delta = np.mean(all_delta)
    # Imprimir os resultados
    print(f"Iteração {idx}:")
    print(f"Erro Relativo Absoluto (absrel): {absrel}")
    print(f"Precisão Delta (measures): {measures}")
print(f"Média do Erro Relativo Absoluto: {mean_absrel}")
print(f"Média da Precisão Delta: {mean_delta}")