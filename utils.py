import cv2
import os
import numpy as np
import glob

def calculate_absrel(gt_path, pred_path):
    # Lista para armazenar os AbsRel calculados de cada imagem
    absrel_scores = []

    # Obter lista de arquivos nos diretórios
    gt_files = glob.glob(os.path.join(gt_path, "*.png"))
    pred_files = glob.glob(os.path.join(pred_path, "*.png"))

    # Garantir que temos a mesma quantidade de arquivos em ambas as pastas
    assert len(gt_files) == len(pred_files)

    # Iterar sobre os arquivos
    for gt_file, pred_file in zip(gt_files, pred_files):
        # Ler imagens de gt e pred em escala de cinza
        gt_image = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)

        # Calcular AbsRel
        absrel = np.mean(np.abs(gt_image - pred_image) / gt_image)
        absrel_scores.append(absrel)

        # Exibir o AbsRel de cada imagem individualmente
        print(f"AbsRel da imagem {os.path.basename(gt_file)}: {absrel}")

    # Calcular a média do AbsRel
    mean_absrel = np.mean(absrel_scores)

    # Exibir a média do AbsRel
    print(f"Média do AbsRel: {mean_absrel}")

def calculate_delta(gt_path, pred_path):
    # Lista para armazenar os Delta calculados de cada imagem
    delta_scores = []

    # Obter lista de arquivos nos diretórios
    gt_files = glob.glob(os.path.join(gt_path, "*.png"))
    pred_files = glob.glob(os.path.join(pred_path, "*.png"))

    # Garantir que temos a mesma quantidade de arquivos em ambas as pastas
    assert len(gt_files) == len(pred_files)

    # Iterar sobre os arquivos
    for gt_file, pred_file in zip(gt_files, pred_files):
        # Ler imagens de gt e pred em escala de cinza
        gt_image = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)

        # Calcular Delta
        max_ratio = np.max(np.maximum(gt_image / pred_image, pred_image / gt_image))
        delta = max_ratio < 1.25
        delta_scores.append(delta)

        # Exibir o Delta de cada imagem individualmente
        print(f"Delta da imagem {os.path.basename(gt_file)}: {delta}")

    # Calcular a média do Delta
    mean_delta = np.mean(delta_scores)

    # Exibir a média do Delta
    print(f"Média do Delta: {mean_delta}")