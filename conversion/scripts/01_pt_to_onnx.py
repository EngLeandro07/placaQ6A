#!/usr/bin/env python3
# =============================================================================
#  01_pt_to_onnx.py
#  Exporta um modelo YOLOv8 (.pt) para ONNX.
#
#  ONDE RODA: no venv de EXPORT (/opt/venv-export/bin/python), que tem o
#  ultralytics isolado. NUNCA rode no Python do QAIRT - o ultralytics puxa
#  numpy>=2 e quebraria o SDK.
#
#  USO:
#     /opt/venv-export/bin/python scripts/01_pt_to_onnx.py
#
#  O bloco CONFIG abaixo tem os valores padrao. Edite-os conforme sua
#  necessidade (caminhos, tamanho de entrada, opset). Nao ha argparse de
#  proposito: a entrada e' este bloco, para ficar explicito e reproduzivel.
# =============================================================================

import shutil
from pathlib import Path
from ultralytics import YOLO


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
# Caminho do modelo treinado (.pt).
# ALTERE para o seu modelo. Padrao aponta para a pasta input-models/ montada.
PT_PATH = "input-models/260420_1280_large.pt"

# Caminho de saida do ONNX. Nome derivado automaticamente do stem de PT_PATH
# (ex.: "modelo_260409m-2.pt" -> "modelo_260409m-2.onnx"), pra diferenciar as
# saidas de modelos diferentes dentro de output-models/.
ONNX_OUT = f"output-models/{Path(PT_PATH).stem}.onnx"

# Resolucao de entrada (lado do quadrado). Default vem de model.env (IMGSZ) -
# edite la' pra manter em sincronia com 02_onnx_to_dlc.py/gen_calibration.py/
# board_test. ALTERE aqui so' se quiser um valor DIFERENTE so' pra este passo.
IMGSZ = int(_shared("IMGSZ", 1280))

# opset ONNX. 11+ e' compativel com QAIRT; YOLOv8 vai bem em 12-17.
# Se o qairt-converter reclamar de operador, tente baixar/subir o opset.
OPSET = 12

# Simplificar o grafo ONNX (recomendado: reduz operadores redundantes).
SIMPLIFY = True

# batch fixo em 1 (NPU embarcada roda 1 imagem por vez).
# dynamic=False -> shapes estaticos, melhor para quantizacao/HTP.
DYNAMIC = False
# =============================================================================


def main():
    pt = Path(PT_PATH)
    if not pt.exists():
        raise FileNotFoundError(
            f"Modelo .pt nao encontrado: {pt}\n"
            f"Coloque seu modelo em '{PT_PATH}' ou edite PT_PATH no CONFIG."
        )

    print(f"[export] carregando {pt}")
    model = YOLO(str(pt))

    print(f"[export] exportando ONNX  imgsz={IMGSZ}  opset={OPSET}  "
          f"simplify={SIMPLIFY}  dynamic={DYNAMIC}")
    # O ultralytics gera o .onnx ao lado do .pt; depois movemos para ONNX_OUT.
    produced = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=SIMPLIFY,
        dynamic=DYNAMIC,
    )

    produced = Path(produced)
    out = Path(ONNX_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out.resolve():
        shutil.move(str(produced), str(out))

    print(f"[export] OK -> {out}")
    print("[export] proximo passo: 02_onnx_to_dlc.py (no Python do QAIRT)")


if __name__ == "__main__":
    main()
