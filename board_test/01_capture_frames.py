#!/usr/bin/env python3
# =============================================================================
#  01_capture_frames.py
#  Captura frames da webcam DA PLACA e grava no mesmo formato .raw usado na
#  calibracao (ver calibration/gen_calibration.py) - resolucao, RGB, NCHW,
#  normalizacao /255. Isso garante que o teste na placa usa exatamente o
#  mesmo pre-processamento que foi usado para calibrar o modelo.
#
#  ONDE RODA: na PLACA, com Python3 do sistema (precisa de opencv-python e
#  numpy instalados na placa - nao faz parte do venv-export do container).
#
#  USO (na placa, dentro de board_test/):
#     python3 01_capture_frames.py
# =============================================================================

import time
from pathlib import Path

import numpy as np
import cv2


def _shared(key, fallback):
    """Le' 'key' de model.env - fonte unica de verdade compartilhada entre
    container/board_test/native_infer (ver esse arquivo na raiz do repo).
    Ao fazer deploy pra placa via scp, copie model.env pra dentro de
    board_test/ tambem (ver README.md) - senao cai no 'fallback' abaixo."""
    for p in (Path(__file__).resolve().parent / "model.env",
              Path("model.env"),
              Path(__file__).resolve().parent.parent / "model.env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith(key + "="):
                    return line.split("=", 1)[1].strip()
    return fallback


# =============================== CONFIG ======================================
# Indice do dispositivo da webcam. NAO assuma 0: em placas com encoder de
# video proprio do SoC (ex.: qcom-venus), /dev/video0/1 podem ser esse
# encoder, nao a webcam. Confira com `v4l2-ctl --list-devices` na placa e
# procure o node com capability "Video Capture" (nao "Metadata Capture").
# Nesta Q6A com uma Logitech C920, o node de captura ficou em /dev/video2.
CAM_INDEX = 2

# Quantos frames capturar. ~100-300 e' suficiente pra ver o comportamento
# (FPS, latencia) sem gerar um dataset gigante. ALTERE conforme necessidade.
NUM_FRAMES = 150

# Pasta de saida dos .raw capturados + input_list.txt.
OUTPUT_DIR = "captures"
INPUT_LIST = "input_list.txt"

# Resolucao - DEVE bater com o IMGSZ usado no treino/export/calibracao do
# modelo que voce vai testar. Default vem de model.env - edite la' (ou copie
# um model.env atualizado pra dentro de board_test/ ao fazer o deploy).
IMGSZ = int(_shared("IMGSZ", 1280))

# Mesmo pre-processamento de calibration/gen_calibration.py: NCHW, RGB, /255,
# sem mean/std. Se o modelo usar outra normalizacao, ajuste aqui E LA para
# ficar identico (senao a saida da NPU nao bate com o que o modelo espera).
LAYOUT = "NCHW"
SCALE = 1.0 / 255.0
MEAN = [0.0, 0.0, 0.0]
STD = [1.0, 1.0, 1.0]
TO_RGB = True
# =============================================================================


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

    if TO_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = img.astype(np.float32)
    x = (x - np.array(MEAN, dtype=np.float32)) * SCALE
    x = x / np.array(STD, dtype=np.float32)

    if LAYOUT == "NCHW":
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
    elif LAYOUT == "NHWC":
        x = np.expand_dims(x, 0)
    else:
        raise ValueError(f"LAYOUT invalido: {LAYOUT}")

    return np.ascontiguousarray(x, dtype=np.float32)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"nao consegui abrir a webcam (indice {CAM_INDEX}). "
            f"confira /dev/video* e se outro processo nao esta usando a camera."
        )

    print(f"[capture] webcam aberta (indice {CAM_INDEX}). "
          f"capturando {NUM_FRAMES} frames, imgsz={IMGSZ}...")

    raw_paths = []
    n_ok = 0
    t0 = time.perf_counter()
    while n_ok < NUM_FRAMES:
        ok, frame = cap.read()
        if not ok:
            print(f"[capture] aviso: falha ao ler frame {n_ok}, tentando de novo")
            continue

        x = preprocess(frame)
        raw_path = out_dir / f"frame_{n_ok:05d}.raw"
        x.tofile(str(raw_path))
        raw_paths.append(str(raw_path))
        n_ok += 1

        if n_ok % 25 == 0:
            print(f"[capture]   {n_ok}/{NUM_FRAMES}")

    cap.release()
    dt = time.perf_counter() - t0

    with open(INPUT_LIST, "w") as f:
        f.write("\n".join(raw_paths) + "\n")

    print(f"[capture] OK -> {len(raw_paths)} frames em {out_dir}/ "
          f"({dt:.1f}s captura+preprocess, {n_ok/dt:.1f} FPS de captura)")
    print(f"[capture] input_list -> {INPUT_LIST}")
    print("[capture] proximo passo: 02_run_inference.py")


if __name__ == "__main__":
    main()
