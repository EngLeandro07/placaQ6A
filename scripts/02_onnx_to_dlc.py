#!/usr/bin/env python3
# =============================================================================
#  02_onnx_to_dlc.py
#  Converte um ONNX em DLC (float, ainda nao quantizado) via qairt-converter.
#
#  ONDE RODA: no Python do QAIRT (o do SDK), NAO no venv-export. Este passo usa
#  o qairt-converter, que faz parte do SDK. Por isso aqui nao se importa
#  ultralytics/torch - so' chamamos a ferramenta de linha de comando do QAIRT.
#
#  USO (dentro do container, ambiente QAIRT ativo):
#     python3 scripts/02_onnx_to_dlc.py
#
#  Edite o CONFIG conforme necessario.
# =============================================================================

import subprocess
import sys
from pathlib import Path

# =============================== CONFIG ======================================
# ONNX de entrada (saida do passo 01).
ONNX_IN = "workspace/models/modelo.onnx"

# DLC float de saida.
DLC_OUT = "workspace/models/modelo_fp.dlc"

# Nome e shape do tensor de entrada do seu modelo.
# Para YOLOv8 exportado NCHW: nome "images", shape 1,3,IMGSZ,IMGSZ.
# CONFIRA o nome real abrindo o ONNX (ex.: com netron) e ALTERE se diferente.
# ALTERE o 640 para a resolucao do seu modelo.
INPUT_NAME = "images"
INPUT_SHAPE = "1,3,1280,1280"

# Caminho do qairt-converter. Em geral esta' no PATH apos o envsetup do SDK.
# Se nao estiver, aponte o caminho absoluto (ex.: $QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter).
CONVERTER = "qairt-converter"
# =============================================================================


def main():
    onnx = Path(ONNX_IN)
    if not onnx.exists():
        raise FileNotFoundError(
            f"ONNX nao encontrado: {onnx}\n"
            f"Rode o passo 01 primeiro, ou ajuste ONNX_IN."
        )

    out = Path(DLC_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        CONVERTER,
        "--input_network", str(onnx),
        "--output_path", str(out),
        "-d", INPUT_NAME, INPUT_SHAPE,
    ]

    print("[convert] executando:")
    print("  " + " ".join(cmd))
    # sem captura: deixamos o log do converter aparecer direto no terminal
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("\n[convert] FALHOU. Dicas:", file=sys.stderr)
        print("  - se reclamar de operador nao suportado, tente outro opset "
              "no passo 01 (ex.: 11 ou 13).", file=sys.stderr)
        print("  - confira INPUT_NAME/INPUT_SHAPE: precisam casar com o ONNX.",
              file=sys.stderr)
        sys.exit(ret.returncode)

    print(f"[convert] OK -> {out}  (DLC float, ainda NAO quantizado)")
    print("[convert] proximo passo: 03_quantize_dlc.py")


if __name__ == "__main__":
    main()
