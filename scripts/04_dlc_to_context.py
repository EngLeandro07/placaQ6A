#!/usr/bin/env python3
# =============================================================================
#  04_dlc_to_context.py
#  Gera o CONTEXT-BINARY (.bin) a partir do DLC INT8, via
#  qnn-context-binary-generator. Faz a "preparacao" especifica do HTP do
#  QCS6490 AQUI no host (offline), em vez de na placa - o que reduz tempo de
#  init e memoria na inferencia.
#
#  OPCIONAL? O .dlc INT8 ja roda na placa via qnn-net-run --dlc_path.
#  O context-binary e' uma OTIMIZACAO: pre-compila o grafo para o SoC. Vale a
#  pena para producao. Para um primeiro teste, pode pular e usar o .dlc direto.
#
#  ONDE RODA: no Python do QAIRT (SDK).
#
#  USO:
#     python3 scripts/04_dlc_to_context.py
# =============================================================================

import json
import subprocess
import sys
from pathlib import Path

# =============================== CONFIG ======================================
# DLC INT8 de entrada (saida do passo 03).
DLC_IN = "workspace/models/modelo_int8.dlc"

# Context-binary de saida (.bin) que vai para a placa.
BIN_OUT = "workspace/models/modelo_int8.bin"

# Backend HTP do host (lib x86 do SDK). Em geral no PATH/lib do SDK.
# Se nao resolver, aponte o caminho absoluto da libQnnHtp.so x86 do SDK.
HTP_BACKEND = "libQnnHtp.so"

# ---- Config do SoC: ESTES VALORES SAO ESPECIFICOS DA Q6A / QCS6490 ----
# dsp_arch v68 e soc_id 35 = QCS6490. NAO altere a menos que mude de placa.
DSP_ARCH = "v68"
SOC_ID = 35

# vtcm_mb: 0 deixa o gerador escolher. Ajuste so' se souber o que faz.
VTCM_MB = 0

# Nome do grafo. Por padrao deixamos generico; alguns fluxos exigem casar com
# o nome interno do modelo. Se o gerador reclamar, ajuste.
GRAPH_NAME = "modelo"
# =============================================================================

HTP_CONFIG_PATH = "workspace/htp_config.json"
# --config_file do qnn-context-binary-generator so aceita um wrapper de
# "backend extensions" apontando para o JSON de graphs/devices - nao aceita
# o JSON de graphs/devices diretamente (senao da' "Unknown Key" em cada campo).
HTP_BACKEND_EXT_PATH = "workspace/htp_backend_ext.json"
HTP_EXT_LIB = "libQnnHtpNetRunExtensions.so"
GEN = "qnn-context-binary-generator"


def write_htp_config():
    cfg = {
        "graphs": [{"graph_names": [GRAPH_NAME], "vtcm_mb": VTCM_MB}],
        "devices": [{"dsp_arch": DSP_ARCH, "soc_id": SOC_ID}],
    }
    Path(HTP_CONFIG_PATH).write_text(json.dumps(cfg, indent=2))

    backend_ext_cfg = {
        "backend_extensions": {
            "shared_library_path": HTP_EXT_LIB,
            "config_file_path": HTP_CONFIG_PATH,
        }
    }
    Path(HTP_BACKEND_EXT_PATH).write_text(json.dumps(backend_ext_cfg, indent=2))

    print(f"[ctx] config HTP -> {HTP_CONFIG_PATH}  "
          f"(dsp_arch={DSP_ARCH}, soc_id={SOC_ID})")
    print(f"[ctx] backend extension config -> {HTP_BACKEND_EXT_PATH}")


def main():
    dlc = Path(DLC_IN)
    if not dlc.exists():
        raise FileNotFoundError(
            f"DLC INT8 nao encontrado: {dlc}\nRode o passo 03 primeiro."
        )

    out = Path(BIN_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_htp_config()

    cmd = [
        GEN,
        "--backend", HTP_BACKEND,
        "--dlc_path", str(dlc),
        "--binary_file", out.stem,            # gera <stem>.bin no output_dir
        "--output_dir", str(out.parent),
        "--config_file", HTP_BACKEND_EXT_PATH,
    ]

    print("[ctx] executando:")
    print("  " + " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("\n[ctx] FALHOU. Dicas:", file=sys.stderr)
        print("  - confira que GRAPH_NAME casa com o nome do grafo no DLC.",
              file=sys.stderr)
        print("  - confira o caminho do backend HTP x86 (HTP_BACKEND).",
              file=sys.stderr)
        sys.exit(ret.returncode)

    print(f"[ctx] OK -> {out}")
    print("[ctx] leve este .bin para a placa e rode com:")
    print("      qnn-net-run --backend libQnnHtp.so "
          "--retrieve_context modelo_int8.bin --input_list <lista> "
          "--output_dir out")


if __name__ == "__main__":
    main()
