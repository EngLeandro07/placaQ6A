#!/usr/bin/env bash
# =============================================================================
#  run_test.sh - orquestra o teste de webcam NA PLACA.
#    - passo capture -> captura frames da webcam e grava .raw (Python sistema)
#    - passo infer   -> roda o modelo na NPU via qnn-net-run sobre os frames
#  Requer: opencv-python + numpy instalados no Python da placa, e o runtime
#  QAIRT (env.sh) sourced ANTES de rodar (ou este script tenta source-ar
#  QAIRT_ENV_SH se a variavel estiver definida).
#
#  USO (na placa, dentro de board_test/):
#     source /caminho/para/env.sh     # runtime QAIRT (uma vez por sessao)
#     ./run_test.sh            # captura + roda tudo
#     ./run_test.sh capture    # so' captura os frames da webcam
#     ./run_test.sh infer      # so' roda a inferencia (usa captura existente)
# =============================================================================
set -e

# Usa a venv local (opencv+numpy) se existir; senao cai pro Python do sistema.
# Python 3.12 da placa e' "externally managed" (PEP 668), entao o setup
# esperado e' criar essa venv uma vez (ver README.md) em vez de pip install
# direto no sistema.
if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
else
  PY="python3"
fi

step_capture() {
  echo "=== [1/2] capturando frames da webcam ==="
  "$PY" 01_capture_frames.py
}

step_infer() {
  echo "=== [2/2] rodando inferencia na NPU (qnn-net-run) ==="
  "$PY" 02_run_inference.py
}

case "${1:-all}" in
  capture) step_capture ;;
  infer)   step_infer ;;
  all)
    step_capture
    step_infer
    echo "=== TESTE COMPLETO. Saidas em outputs/. ==="
    ;;
  *)
    echo "uso: $0 [all|capture|infer]"
    exit 1
    ;;
esac
