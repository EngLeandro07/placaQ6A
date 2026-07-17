# Teste de webcam na placa (Q6A)

Ambiente mínimo, feito para rodar **direto na Radxa Dragon Q6A** (não no
container de conversão), que simula um cenário de produção: captura frames de
uma webcam USB conectada à placa e roda o modelo compilado (`.dlc` ou `.bin`)
na NPU via `qnn-net-run`. Sem NMS/decode de detecções — o objetivo é só
confirmar que o modelo carrega e roda na HTP com entrada de câmera real, e
medir latência/FPS. É headless (sem monitor): tudo sai como log no terminal.

## Pré-requisitos na placa

- Runtime QAIRT **2.42** já instalado (mesma versão usada para gerar o
  `.dlc`/`.bin` no host) e o `env.sh` desse runtime disponível.
- Webcam USB conectada (normalmente aparece como `/dev/video0`).
- Python 3 com `opencv-python-headless` (sem monitor, não precisa de GTK/Qt) e
  `numpy`. Python 3.12 costuma vir "externally managed" (PEP 668), então crie
  uma venv isolada em vez de `pip install` direto no sistema:
  ```bash
  cd board_test
  python3 -m venv venv
  venv/bin/pip install --upgrade pip
  venv/bin/pip install opencv-python-headless numpy
  ```
  O `run_test.sh` detecta e usa `venv/bin/python` automaticamente se existir.

## Levar para a placa

**Opção 1 — `board/` montado (recomendado, ver `board_mount.sh` na raiz):**
```bash
./board_mount.sh mount        # uma vez por sessão, monta ~/mctech em ./board/
cp -r board_test model.env board/
cp workspace/models/modelo_int8.bin board/board_test/
# ou, se for testar o .dlc em vez do .bin:
# cp workspace/models/modelo_int8.dlc board/board_test/
```
Como `board/` é o mesmo filesystem da placa (via `sshfs`), esses `cp` já
escrevem direto lá — nada de `scp` repetido, e dá pra editar os scripts em
`board_test/` no host e ver o efeito na placa na hora.

**Opção 2 — `scp` direto (se não tiver o mount configurado):**
```bash
scp -r board_test radxa@<ip-da-placa>:~/mctech/board_test
scp model.env radxa@<ip-da-placa>:~/mctech/board_test/
scp workspace/models/modelo_int8.bin radxa@<ip-da-placa>:~/mctech/board_test/
```

O `model.env` (raiz do repo) é a fonte única de verdade de valores como
`IMGSZ` — copie-o de novo sempre que ele mudar no host. Sem ele, os scripts
caem no fallback hardcoded no próprio `CONFIG` (pode divergir do que foi
usado pra gerar o modelo!). Ver `CLAUDE.md`, seção "Environments & shared
data".

## Rodar na placa

```bash
source /caminho/para/qairt/2.42.x/lib/../envsetup.sh   # ou o env.sh do runtime
cd ~/mctech/board_test
chmod +x run_test.sh
./run_test.sh              # captura webcam + roda inferência
```

Passo a passo (útil para depurar):
```bash
./run_test.sh capture   # só captura frames da webcam -> captures/ + input_list.txt
./run_test.sh infer     # só roda qnn-net-run sobre a captura existente -> outputs/
```

## Configuração

Cada script tem um bloco `CONFIG` no topo (mesmo padrão dos scripts do
pipeline principal, sem `argparse`):

- `01_capture_frames.py`: `CAM_INDEX`, `NUM_FRAMES`, `IMGSZ` (default lido de
  `model.env`) e o pré-processamento (deve ser **idêntico** ao usado em
  `calibration/gen_calibration.py` — resolução, RGB, NCHW, `/255`).
- `02_run_inference.py`: `MODEL_MODE` (`"bin"` ou `"dlc"`), caminhos do
  modelo, e o `BACKEND` (`$QNN_SDK_ROOT`/`$VARIANT`, resolvidos do `env.sh`).

## Lendo o resultado

O `02_run_inference.py` imprime, ao final:
- tempo total e latência média/FPS amortizado (inclui o load do modelo 1x);
- estatísticas (min/max/mean) do primeiro e do último tensor de saída
  gerados, só para confirmar que a saída não é lixo/zero/NaN.

Se quiser inspecionar as detecções de verdade, os `.raw` de saída ficam em
`outputs/Result_N/` — decodifique com o mesmo código de pós-processamento do
seu YOLOv8 (grid/anchors, sigmoid, NMS), fora do escopo deste teste rápido.
