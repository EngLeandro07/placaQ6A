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
              Path(__file__).resolve().parent.parent / "model.env",
              Path(__file__).resolve().parent.parent.parent / "model.env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith(key + "="):
                    return line.split("=", 1)[1].strip()
    return fallback


# =============================== CONFIG ======================================
# DLC INT8 de entrada (saida do passo 03). ATUALIZE junto com DLC_OUT do
# passo 03 ao trocar de modelo.
DLC_IN = "output-models/260420_1280_large_int8.dlc"

_dlc_in_stem = Path(DLC_IN).stem
_base_name = _dlc_in_stem[:-5] if _dlc_in_stem.endswith("_int8") else _dlc_in_stem

# Context-binary de saida (.bin) que vai para a placa. Nome derivado do stem
# de DLC_IN (removendo o sufixo "_int8") - sem sufixo adicional, ja que e' o
# unico .bin gerado (a extensao ja diferencia de .dlc/.onnx).
BIN_OUT = f"output-models/{_base_name}.bin"

# Backend HTP do host (lib x86 do SDK). Em geral no PATH/lib do SDK.
# Se nao resolver, aponte o caminho absoluto da libQnnHtp.so x86 do SDK.
HTP_BACKEND = "libQnnHtp.so"

# ---- Config do SoC: defaults vem de model.env. SO' muda se trocar de placa,
# NAO ao trocar de modelo. ----
DSP_ARCH = _shared("DSP_ARCH", "v68")
SOC_ID = int(_shared("SOC_ID", 35))

# vtcm_mb: quanto de VTCM (memoria on-chip rapida, dedicada ao HTP) reservar
# pro grafo. Mais VTCM = menos spill pra DDR = mais rapido, mas exige mais
# memoria DMA reservada no device tree da placa pra alocar via FastRPC.
#
# BUG CONHECIDO nesta placa (imagem Radxa R2, ver memoria do projeto / topico
# no forum): o device tree tem uma reserva de DMA pro FastRPC/CDSP bem menor
# que o esperado (faltam nos `memory-region` nos `compute-cb@N`). Testado
# empiricamente no board (2026-07-20): vtcm_mb <= 2 FUNCIONA (roda de verdade
# na NPU), vtcm_mb >= 3 FALHA com "Request feature vtcm size with value ...
# unsupported" / err 0x138d. Ou seja, 2 e' o teto que esta placa aceita hoje -
# nao e' um limite do modelo nem do QAIRT, e' especifico deste bug de imagem.
# Se a Radxa corrigir o device tree numa imagem futura, provavelmente da' pra
# subir esse valor de novo (4 ou 8) e ganhar performance (menos spill pra
# DDR) - vale re-testar depois de qualquer atualizacao de imagem/firmware.
VTCM_MB = 2

# Nome do grafo. Default vem de model.env (GRAPH_NAME) - NAO e' livre: o
# qairt-converter usa o stem do DLC_OUT do passo 02 como nome do grafo (ver
# comentario em model.env), e a quantizacao (passo 03) preserva esse nome no
# INT8. O fallback abaixo (usado so' se a chave faltar em model.env) assume
# o padrao "<base>_fp" derivado do proprio DLC_IN. Se o gerador/native_infer
# reclamar que o nome nao bate, confira o valor real com native_infer/qnn_infer
# (lista os grafos do .bin) e atualize model.env.
GRAPH_NAME = _shared("GRAPH_NAME", f"{_base_name}_fp")
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
    print(f"      qnn-net-run --backend libQnnHtp.so "
          f"--retrieve_context {out.name} --input_list <lista> "
          f"--output_dir out")


if __name__ == "__main__":
    main()
