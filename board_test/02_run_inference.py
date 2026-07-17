#!/usr/bin/env python3
# =============================================================================
#  02_run_inference.py
#  Roda o modelo compilado (.dlc ou .bin) na NPU (HTP) da placa sobre os
#  frames capturados no passo 01, via qnn-net-run (mesma ferramenta usada
#  para testar manualmente, ver README.md). Nao decodifica as deteccoes do
#  YOLO (NMS etc.) de proposito - o objetivo aqui e' so confirmar que o
#  modelo CARREGA e RODA na NPU com entrada real de camera, e medir
#  latencia/FPS. Para inspecionar deteccoes, leia os .raw de saida a mao.
#
#  ONDE RODA: na PLACA, com o runtime QAIRT ja carregado (source env.sh),
#  que deixa qnn-net-run no PATH e $QNN_SDK_ROOT / $VARIANT exportados.
#
#  USO (na placa, dentro de board_test/, com env.sh ja sourced):
#     python3 02_run_inference.py
# =============================================================================

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# =============================== CONFIG ======================================
# Qual artefato testar: "bin" (context-binary, --retrieve_context) ou
# "dlc" (--dlc_path). O .bin carrega mais rapido (ja pre-compilado pro SoC).
MODEL_MODE = "bin"

DLC_PATH = "modelo_int8.dlc"
BIN_PATH = "modelo_int8.bin"

# input_list.txt gerado pelo passo 01.
INPUT_LIST = "input_list.txt"

# Onde qnn-net-run escreve as saidas (um Result_N/ por linha do input_list).
OUTPUT_DIR = "outputs"

# Backend HTP da PLACA (arm64). $QNN_SDK_ROOT e $VARIANT vem do env.sh do
# runtime QAIRT - se nao estiverem setados, rode `source env.sh` antes.
BACKEND = "$QNN_SDK_ROOT/lib/$VARIANT/libQnnHtp.so"

GEN = "qnn-net-run"
# =============================================================================


def resolve_backend() -> str:
    backend = os.path.expandvars(BACKEND)
    if "$" in backend:
        raise EnvironmentError(
            f"variavel de ambiente nao resolvida em BACKEND: {backend}\n"
            f"rode 'source env.sh' do runtime QAIRT antes deste script."
        )
    return backend


def main():
    ilist = Path(INPUT_LIST)
    if not ilist.exists():
        raise FileNotFoundError(
            f"{ilist} nao encontrado. rode 01_capture_frames.py primeiro."
        )
    n_frames = sum(1 for _ in ilist.open())

    if MODEL_MODE == "bin":
        model_flag, model_path = "--retrieve_context", BIN_PATH
    elif MODEL_MODE == "dlc":
        model_flag, model_path = "--dlc_path", DLC_PATH
    else:
        raise ValueError(f"MODEL_MODE invalido: {MODEL_MODE!r} (use 'bin' ou 'dlc')")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"modelo nao encontrado: {model_path}\n"
            f"copie o artefato (scp) pra dentro de board_test/ antes de rodar."
        )

    backend = resolve_backend()
    out_dir = Path(OUTPUT_DIR)

    cmd = [
        GEN,
        "--backend", backend,
        model_flag, model_path,
        "--input_list", str(ilist),
        "--output_dir", str(out_dir),
    ]

    print("[infer] executando:")
    print("  " + " ".join(cmd))
    print(f"[infer] {n_frames} frames no input_list, modo={MODEL_MODE}")

    t0 = time.perf_counter()
    ret = subprocess.run(cmd)
    dt = time.perf_counter() - t0

    if ret.returncode != 0:
        print("\n[infer] FALHOU. Dicas:", file=sys.stderr)
        print("  - confira se 'source env.sh' do runtime QAIRT foi feito.",
              file=sys.stderr)
        print("  - confira se o modelo foi compilado com a mesma versao "
              "QAIRT do runtime da placa (2.42 dos dois lados).",
              file=sys.stderr)
        sys.exit(ret.returncode)

    print(f"[infer] OK em {dt:.2f}s total (inclui carregar o modelo 1x) "
          f"-> {dt/n_frames*1000:.1f} ms/frame medio, "
          f"{n_frames/dt:.1f} FPS medio (amortizado)")

    summarize_outputs(out_dir, n_frames)


def summarize_outputs(out_dir: Path, n_frames: int):
    result_dirs = sorted(out_dir.glob("Result_*"), key=lambda p: p.stat().st_mtime)
    if not result_dirs:
        print(f"[infer] aviso: nenhum Result_N encontrado em {out_dir}/ "
              f"(confira se qnn-net-run rodou corretamente)")
        return

    print(f"[infer] {len(result_dirs)} resultados em {out_dir}/. "
          f"amostra do primeiro e do ultimo:")

    for label, rdir in (("primeiro", result_dirs[0]), ("ultimo", result_dirs[-1])):
        raw_files = sorted(rdir.glob("*.raw"))
        if not raw_files:
            print(f"  [{label}] {rdir.name}: nenhum .raw de saida encontrado")
            continue
        for rf in raw_files:
            arr = np.fromfile(str(rf), dtype=np.float32)
            print(f"  [{label}] {rdir.name}/{rf.name}: "
                  f"{arr.size} floats  min={arr.min():.4f}  "
                  f"max={arr.max():.4f}  mean={arr.mean():.4f}")

    print("[infer] se os numeros acima parecem plausiveis (nao sao tudo "
          "zero/NaN/constante), o modelo esta rodando corretamente na NPU.")


if __name__ == "__main__":
    main()
