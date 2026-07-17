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


def _shared(key, fallback):
    """Le' 'key' de model.env (raiz do repo, fonte unica de verdade
    compartilhada entre container/board_test/native_infer - ver esse
    arquivo). Se nao encontrar o arquivo/chave, usa 'fallback'."""
    for p in (Path(__file__).resolve().parent / "model.env",
              Path("model.env"),
              Path(__file__).resolve().parent.parent / "model.env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith(key + "="):
                    return line.split("=", 1)[1].strip()
    return fallback


# =============================== CONFIG ======================================
# DLC INT8 de entrada (saida do passo 03).
DLC_IN = "workspace/models/modelo_int8.dlc"

# Context-binary de saida (.bin) que vai para a placa.
BIN_OUT = "workspace/models/modelo_int8.bin"

# Backend HTP do host (lib x86 do SDK). Em geral no PATH/lib do SDK.
# Se nao resolver, aponte o caminho absoluto da libQnnHtp.so x86 do SDK.
HTP_BACKEND = "libQnnHtp.so"

# ---- Config do SoC: defaults vem de model.env. SO' muda se trocar de placa,
# NAO ao trocar de modelo. ----
DSP_ARCH = _shared("DSP_ARCH", "v68")
SOC_ID = int(_shared("SOC_ID", 35))

# vtcm_mb: 0 deixa o gerador escolher. Na Q6A, o .bin gerado (com 0 OU 8)
# falha igual em runtime ("Request feature vtcm size with value 4194304
# unsupported") - testado com 1280x1280 E 640x640, mesmo erro nos dois.
# Ou seja, ESTE campo nao e' a causa do problema (o valor pedido em runtime
# nao mudou ao alterar vtcm_mb aqui). A causa provavel e' incompatibilidade
# de firmware/skeleton do DSP na placa com esta versao do QAIRT - nao um
# ajuste de config do lado do host. Deixado em 8 so' por nao ter piorado nada.
VTCM_MB = 8

# Nome do grafo. Default vem de model.env (GRAPH_NAME) - NAO e' livre: o
# qairt-converter usa o stem do DLC_OUT do passo 02 como nome do grafo (ver
# comentario em model.env). Se o gerador/native_infer reclamar que o nome
# nao bate, confira o valor real com native_infer/qnn_infer (lista os grafos
# do .bin) e atualize model.env.
GRAPH_NAME = _shared("GRAPH_NAME", "modelo_fp")
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
