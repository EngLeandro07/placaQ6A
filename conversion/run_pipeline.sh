#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh - orquestra o pipeline completo dentro do container.
#  Roda cada passo no Python CORRETO:
#    - passo 01 e calibracao -> venv-export (ultralytics, numpy>=2)
#    - passos 02, 03, 04      -> Python do QAIRT (SDK, numpy<2)
#  E' isso que mantem o isolamento que evita o conflito de dependencias.
#
#  USO (dentro do container, a partir de /workspace):
#     ./run_pipeline.sh            # roda tudo
#     ./run_pipeline.sh export     # so' .pt -> .onnx
#     ./run_pipeline.sh calib      # so' gerar dataset de calibracao
#     ./run_pipeline.sh convert    # so' .onnx -> .dlc
#     ./run_pipeline.sh quant      # so' quantizar
#     ./run_pipeline.sh context    # so' gerar context-binary
# =============================================================================
set -e

# Python do venv de export (isolado).
PY_EXPORT="/opt/venv-export/bin/python"

# Python do QAIRT. O envsetup do SDK normalmente deixa 'python3' apontando
# para o ambiente certo e as ferramentas no PATH. Se o seu SDK usa outro
# binario, ALTERE aqui.
PY_QAIRT="python3"

# Carrega o ambiente do QAIRT (envsetup). AJUSTE o caminho conforme a imagem.
# Na imagem da Radxa o SDK costuma estar em /root/qairt/<versao> ou similar.
# Descubra com:  find / -name envsetup.sh 2>/dev/null
QAIRT_ENVSETUP="${QAIRT_ENVSETUP:-/root/qairt/2.42.0.251225/bin/envsetup.sh}"

step_export() {
  echo "=== [1/4] export .pt -> .onnx (venv-export) ==="
  "$PY_EXPORT" scripts/01_pt_to_onnx.py
}

step_calib() {
  echo "=== [calib] dataset -> dataset de calibracao (venv-export) ==="
  "$PY_EXPORT" calibration/gen_calibration.py
}

step_convert() {
  echo "=== [2/4] .onnx -> .dlc float (QAIRT) ==="
  # shellcheck disable=SC1090
  source "$QAIRT_ENVSETUP"
  "$PY_QAIRT" scripts/02_onnx_to_dlc.py
}

step_quant() {
  echo "=== [3/4] quantizacao INT8 (QAIRT) ==="
  # shellcheck disable=SC1090
  source "$QAIRT_ENVSETUP"
  "$PY_QAIRT" scripts/03_quantize_dlc.py
}

step_context() {
  echo "=== [4/4] context-binary (QAIRT) ==="
  # shellcheck disable=SC1090
  source "$QAIRT_ENVSETUP"
  "$PY_QAIRT" scripts/04_dlc_to_context.py
}

case "${1:-all}" in
  export)  step_export ;;
  calib)   step_calib ;;
  convert) step_convert ;;
  quant)   step_quant ;;
  context) step_context ;;
  all)
    step_export
    step_calib
    step_convert
    step_quant
    step_context
    echo "=== PIPELINE COMPLETO. Saidas em output-models/. ==="
    ;;
  *)
    echo "uso: $0 [all|export|calib|convert|quant|context]"
    exit 1
    ;;
esac
