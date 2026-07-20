#!/usr/bin/env python3
# =============================================================================
#  03_quantize_dlc.py
#  Quantiza o DLC float para INT8 usando o dataset de calibracao.
#  O HTP do QCS6490 e' integer-only: este passo e' OBRIGATORIO para a NPU.
#
#  ONDE RODA: no Python do QAIRT (SDK), igual ao passo 02.
#
#  ATENCAO AS FLAGS (mudaram nas versoes recentes do QAIRT):
#    - NAO existe mais --quant_scheme (flag antiga/invalida no 2.42).
#    - O correto agora e' --act_quantizer_calibration / --param_quantizer_calibration
#      (metodo de calibracao) e, opcionalmente, --act_quantizer_schema /
#      --param_quantizer_schema (asymmetric/symmetric).
#
#  USO:
#     python3 scripts/03_quantize_dlc.py
# =============================================================================

import subprocess
import sys
from pathlib import Path

# =============================== CONFIG ======================================
# DLC float de entrada (saida do passo 02). ATUALIZE junto com DLC_OUT do
# passo 02 ao trocar de modelo.
DLC_IN = "output-models/260420_1280_large_fp.dlc"

# DLC quantizado de saida (este e' o que vai para a placa / context-binary).
# Nome derivado do stem de DLC_IN (removendo o sufixo "_fp", se presente) com
# sufixo "_int8" pra diferenciar do float (mesma extensao .dlc).
_dlc_in_stem = Path(DLC_IN).stem
_base_name = _dlc_in_stem[:-3] if _dlc_in_stem.endswith("_fp") else _dlc_in_stem
DLC_OUT = f"output-models/{_base_name}_int8.dlc"

# input_list.txt do dataset de calibracao (saida de gen_calibration.py).
INPUT_LIST = "calibration/input_list.txt"

# Metodo de calibracao das ATIVACOES.
#   min-max (default) | sqnr | entropy | mse | percentile
# min-max e' robusto e padrao. Para modelos com outliers, 'percentile' ou 'mse'
# podem dar acuracia melhor. ALTERE para experimentar.
ACT_CALIB = "min-max"

# Metodo de calibracao dos PARAMETROS (pesos).
PARAM_CALIB = "min-max"

# Bitwidths. INT8 e' o alvo do HTP v68. 8/8/8 e' o padrao.
# Para mais precisao em camadas sensiveis, INT16 e' possivel (act_bitwidth 16),
# mas custa desempenho. ALTERE com cuidado.
ACT_BW = 8
WEIGHTS_BW = 8
BIAS_BW = 8

# Quantizacao por canal nos pesos: melhora acuracia em convs. Recomendado True.
USE_PER_CHANNEL = True

# Caminho do qairt-quantizer (geralmente no PATH apos envsetup).
QUANTIZER = "qairt-quantizer"
# =============================================================================


def main():
    dlc = Path(DLC_IN)
    if not dlc.exists():
        raise FileNotFoundError(
            f"DLC float nao encontrado: {dlc}\nRode o passo 02 primeiro."
        )
    ilist = Path(INPUT_LIST)
    if not ilist.exists():
        raise FileNotFoundError(
            f"input_list de calibracao nao encontrado: {ilist}\n"
            f"Rode calibration/gen_calibration.py primeiro."
        )

    out = Path(DLC_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        QUANTIZER,
        "--input_dlc", str(dlc),
        "--output_dlc", str(out),
        "--input_list", str(ilist),
        "--act_quantizer_calibration", ACT_CALIB,
        "--param_quantizer_calibration", PARAM_CALIB,
        "--act_bitwidth", str(ACT_BW),
        "--weights_bitwidth", str(WEIGHTS_BW),
        "--bias_bitwidth", str(BIAS_BW),
    ]
    if USE_PER_CHANNEL:
        cmd.append("--use_per_channel_quantization")

    print("[quant] executando:")
    print("  " + " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("\n[quant] FALHOU. Dicas:", file=sys.stderr)
        print("  - confira que os .raw do input_list batem com o shape de "
              "entrada do modelo (resolucao/layout/normalizacao).",
              file=sys.stderr)
        print("  - se a flag for rejeitada, rode 'qairt-quantizer --help' e "
              "confira os nomes na SUA versao do SDK.", file=sys.stderr)
        sys.exit(ret.returncode)

    print(f"[quant] OK -> {out}  (DLC INT8, pronto para a NPU)")
    print("[quant] proximo passo: 04_dlc_to_context.py (opcional, recomendado)")


if __name__ == "__main__":
    main()
