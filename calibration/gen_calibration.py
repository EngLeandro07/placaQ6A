#!/usr/bin/env python3
# =============================================================================
#  gen_calibration.py   (mora em calibration/)
#  Converte um dataset de imagens em um DATASET DE CALIBRACAO no formato que o
#  qairt-quantizer consome: arquivos .raw (float32) + um input_list.txt.
#
#  ENTRADA : um diretorio de imagens (jpg/png).
#  SAIDA   : diretorio com <nome>.raw para cada imagem + input_list.txt
#            apontando para os .raw (caminhos que o quantizer vai ler).
#
#  ONDE RODA: pode rodar no venv de EXPORT (tem opencv/numpy). O importante e'
#  que o pre-processamento aqui seja IDENTICO ao que a placa fara na inferencia
#  (mesma resolucao, mesma normalizacao, mesma ordem de canais). Se divergir, a
#  quantizacao calibra errado e a acuracia despenca.
#
#  USO:
#     /opt/venv-export/bin/python calibration/gen_calibration.py
# =============================================================================

from pathlib import Path
import numpy as np
import cv2

# =============================== CONFIG ======================================
# Dataset de ENTRADA (pasta com imagens reais do seu dominio - as fairings).
# ALTERE para a pasta do seu dataset.
INPUT_DIR = "calibration/dataset"

# Diretorio de SAIDA do dataset de calibracao (.raw + input_list.txt).
OUTPUT_DIR = "calibration/calib_raw"

# Nome do input_list que o qairt-quantizer vai consumir.
INPUT_LIST = "calibration/input_list.txt"

# Resolucao - DEVE bater com IMGSZ usado no export e na inferencia da placa.
IMGSZ = 1280

# Layout do tensor de saida:
#   "NCHW" -> (1,3,H,W)  padrao para ONNX/QAIRT exportado do PyTorch.
#   "NHWC" -> (1,H,W,3)  use se seu grafo espera channel-last.
# ALTERE conforme o input do seu modelo (cheque com o passo onnx).
LAYOUT = "NCHW"

# Normalizacao. YOLOv8 espera pixels em [0,1] (divide por 255), sem mean/std.
# Se seu modelo usa outra normalizacao, ajuste aqui PARA BATER COM O TREINO.
SCALE = 1.0 / 255.0
MEAN = [0.0, 0.0, 0.0]   # subtrai antes de escalar (em 0-255). Padrao: nada.
STD = [1.0, 1.0, 1.0]    # divide depois de escalar. Padrao: nada.

# Ordem de canais. OpenCV le BGR; YOLO treina em RGB -> normalmente converte.
# ALTERE para False se seu pipeline usa BGR direto.
TO_RGB = True

# Quantas imagens usar na calibracao. ~200-500 e' um bom intervalo.
# 0 = usar todas as imagens da pasta. ALTERE conforme tamanho do dataset.
MAX_IMAGES = 0

# Extensoes aceitas.
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# =============================================================================


def preprocess(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR, HxWx3, uint8
    if img is None:
        raise ValueError(f"falha ao ler imagem: {img_path}")

    img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

    if TO_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = img.astype(np.float32)
    # normalizacao: (pixel - mean) entao * scale entao / std
    x = (x - np.array(MEAN, dtype=np.float32)) * SCALE
    x = x / np.array(STD, dtype=np.float32)

    if LAYOUT == "NCHW":
        x = np.transpose(x, (2, 0, 1))   # HWC -> CHW
        x = np.expand_dims(x, 0)         # -> 1,C,H,W
    elif LAYOUT == "NHWC":
        x = np.expand_dims(x, 0)         # -> 1,H,W,C
    else:
        raise ValueError(f"LAYOUT invalido: {LAYOUT}")

    return np.ascontiguousarray(x, dtype=np.float32)


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    list_path = Path(INPUT_LIST)

    if not in_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset de entrada nao encontrado: {in_dir}\n"
            f"Coloque imagens em '{INPUT_DIR}' ou edite INPUT_DIR no CONFIG."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in EXTS])
    if not images:
        raise RuntimeError(f"nenhuma imagem ({EXTS}) em {in_dir}")
    if MAX_IMAGES > 0:
        images = images[:MAX_IMAGES]

    print(f"[calib] {len(images)} imagens  imgsz={IMGSZ}  layout={LAYOUT}")

    raw_paths = []
    for i, img_path in enumerate(images):
        x = preprocess(img_path)
        raw_name = f"{img_path.stem}.raw"
        raw_path = out_dir / raw_name
        x.tofile(str(raw_path))           # grava float32 cru
        raw_paths.append(str(raw_path))
        if (i + 1) % 50 == 0:
            print(f"[calib]   {i+1}/{len(images)}")

    # input_list.txt: um caminho .raw por linha. O quantizer le este arquivo.
    with open(list_path, "w") as f:
        f.write("\n".join(raw_paths) + "\n")

    print(f"[calib] OK -> {out_dir}  ({len(raw_paths)} arquivos .raw)")
    print(f"[calib] input_list -> {list_path}")
    print("[calib] use este input_list no passo 03 (quantizacao).")


if __name__ == "__main__":
    main()
