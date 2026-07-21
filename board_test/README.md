# Teste de webcam na placa (Q6A)

Ambiente mínimo, feito para rodar **direto na Radxa Dragon Q6A** (não no
container de conversão), que simula um cenário de produção: captura frames de
uma webcam USB conectada à placa e roda o modelo compilado (`.dlc` ou `.bin`)
na NPU via `qnn-net-run`. Sem NMS/decode de detecções — o objetivo é só
confirmar que o modelo carrega e roda na HTP com entrada de câmera real, e
medir latência/FPS. É headless (sem monitor): tudo sai como log no terminal.

Dois modos de uso, conforme o objetivo:
- **`run_test.sh`** (`01_capture_frames.py` + `02_run_inference.py`): smoke
  test em lote — captura N frames, depois roda `qnn-net-run` uma vez sobre
  todos eles. Rápido de rodar/depurar, bom pra confirmar que o modelo
  carrega e roda.
- **`03_live_loop.py`**: laço **contínuo** de captura+inferência (câmera liga
  uma vez, inferência roda em loop, como em produção de verdade), com
  visualização ao vivo no navegador via `streamer/` e um CSV frame-a-frame —
  ver seção própria abaixo.

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
mkdir -p board/models   # ~/mctech/models/ - COMPARTILHADO com native_infer/, não fica dentro de board_test/
cp conversion/output-models/modelo_int8.bin board/models/
# ou, se for testar o .dlc em vez do .bin:
# cp conversion/output-models/modelo_int8.dlc board/models/
```
Como `board/` é o mesmo filesystem da placa (via `sshfs`), esses `cp` já
escrevem direto lá — nada de `scp` repetido, e dá pra editar os scripts em
`board_test/` no host e ver o efeito na placa na hora.

**Opção 2 — `scp` direto (se não tiver o mount configurado):**
```bash
scp -r board_test radxa@<ip-da-placa>:~/mctech/board_test
scp model.env radxa@<ip-da-placa>:~/mctech/board_test/
ssh radxa@<ip-da-placa> mkdir -p ~/mctech/models
scp conversion/output-models/modelo_int8.bin radxa@<ip-da-placa>:~/mctech/models/
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
  `model.env`), layout NCHW/RGB, uint8 bruto (0-255) — **não** é o mesmo
  formato de `conversion/calibration/gen_calibration.py` (que gera float32
  normalizado, correto só pra calibrar o `.dlc` float no passo 03; o grafo
  INT8 já implantado espera pixel bruto, ver comentário no CONFIG do script).
- `02_run_inference.py`: `MODEL_MODE` (`"bin"` ou `"dlc"`), caminhos do
  modelo em `../models/` (`DLC_PATH`/`BIN_PATH` — diretório **compartilhado**
  com `native_infer/`, ver seção acima), `OUTPUT_DTYPE` (uint8 por padrão) e
  o `BACKEND` (`$QNN_SDK_ROOT`/`$VARIANT`, resolvidos do `env.sh`).

## Lendo o resultado

O `02_run_inference.py` imprime, ao final:
- tempo total e latência média/FPS amortizado (inclui o load do modelo 1x);
- estatísticas (min/max/mean) do primeiro e do último tensor de saída
  gerados, só para confirmar que a saída não é lixo/zero/NaN.

Se quiser inspecionar as detecções de verdade, os `.raw` de saída ficam em
`outputs/Result_N/` — decodifique com o mesmo código de pós-processamento do
seu YOLOv8 (grid/anchors, sigmoid, NMS), fora do escopo deste teste rápido.

## `03_live_loop.py` — captura+inferência contínua, com streamer ao vivo

Laço **contínuo** (não em lote): captura 1 frame → envia pro `streamer/`
(visualização ao vivo) → preprocessa → roda 1 inferência via
`native_infer/qnn_infer --loop` (subprocess persistente, modelo carregado
uma única vez, não a cada frame) → grava 1 linha no CSV → repete, até
Ctrl+C. Simula uso de produção de verdade (câmera liga uma vez, inferência
roda em loop) — `run_test.sh` roda em lote (captura tudo, só depois infere
tudo).

**Pré-requisitos, além dos já listados acima:**
- `native_infer/qnn_infer` compilado na placa (`cd native_infer && make` —
  ver `native_infer/README.md`).
- `streamer/streamer.py` rodando (em outro terminal/sessão SSH) se você
  quiser visualização ao vivo — é opcional (best-effort): sem ele,
  `03_live_loop.py` roda normalmente, só não há imagem no navegador. Ver
  `streamer/README.md`.

```bash
source ../qairt_runtime/env.sh
cd ~/mctech/board_test
python3 ../streamer/streamer.py &   # opcional, em background (ou noutro terminal) - ver streamer/README.md
python3 03_live_loop.py             # Ctrl+C encerra de forma limpa
```

No navegador do host: `http://<ip-da-placa>:8080/`. O CSV frame-a-frame
sai em `experiments/live_<MODEL_NAME>.csv` (colunas: timestamp, índice do
frame, FPS instantâneo, tempos de cada etapa em ms, status da inferência,
min/max/mean da saída) — acompanhe CPU/RAM/temperatura/FPS ao vivo com
`./monitor.sh` (ver seção abaixo) rodando num terceiro terminal.

Edite `MODEL_NAME`/`BIN_PATH`/`GRAPH_NAME` no bloco `CONFIG` do topo do
script ao trocar de modelo (mesmo padrão dos outros scripts deste
diretório).

## Ferramentas de observação (`benchmark_batch.sh`, `monitor.sh`, `plot_results.py`)

Dois usos distintos, não confunda:

- **`benchmark_batch.sh`**: gera **dados** — roda inferência repetida em
  lote e grava um CSV por modelo, pra comparar modelos entre si depois
  (`plot_results.py`).
- **`monitor.sh`**: **não gera nada** — é um visualizador ao vivo (tipo
  `top`) de CPU/RAM/temperatura/FPS, pra rodar num terminal separado
  enquanto `03_live_loop.py` roda em outro.

### 1. Na placa: `benchmark_batch.sh` — benchmark em lote entre modelos

Roda inferência repetida (`ITERATIONS` vezes) sobre um lote de frames
capturado 1x (isola o custo da NPU do custo de captura de webcam), e grava
FPS + CPU% + RAM + temperatura (CPU e NPU/DSP) + estado do CDSP a cada
iteração num CSV. **Extraído do antigo `monitor.sh`** — mesma lógica, sem
mudança de comportamento (`monitor.sh` agora é só o visualizador ao vivo
abaixo).

```bash
source ../qairt_runtime/env.sh
cd ~/mctech/board_test
# edite MODEL_NAME/BIN_PATH no topo do benchmark_batch.sh pro modelo que vai testar
./benchmark_batch.sh
```

Repita trocando `MODEL_NAME`/`BIN_PATH` pra cada modelo (`modelo_260409m-2`,
`260417_1280_nano`, `260420_1280_large`) — cada rodada gera um
`experiments/<MODEL_NAME>.csv` separado. **Lembre de ajustar `IMGSZ` em
`model.env`** antes de cada modelo (960 pro médio, 1280 pros outros dois) —
o `benchmark_batch.sh` recaptura o lote de frames a cada execução, então
usa o `IMGSZ` que estiver configurado no momento.

**Não existe um "% de uso da NPU" exposto por esta placa** (só a GPU tem
`devfreq` com load — o CDSP só tem um contador de tempo em baixo-consumo em
`debugfs`, com unidade não confirmada, não usado pra não inventar número
enganoso). Como proxy de atividade da NPU, o script usa a temperatura da
zona térmica `nspss` (NSP SubSystem = o próprio bloco Hexagon/HTP) e o
estado do remoteproc `cdsp` (detecta crash/recuperação, como o bug de
DMA/VTCM já visto antes, se acontecer de novo).

**Só cobre este ambiente (`board_test` via `qnn-net-run`), não o
`native_infer/`**: esse formato de "N iterações sobre o mesmo lote" não faz
sentido pro `native_infer/qnn_infer --loop` (que já existe, ver
`native_infer/README.md`), desenhado pra uso contínuo real (frame a frame,
ao vivo) — pra medir desempenho do `native_infer`, use `03_live_loop.py`
(que já usa `qnn_infer --loop`) em vez de `benchmark_batch.sh`.

### 1b. Na placa, em paralelo: `monitor.sh` — visualização ao vivo

```bash
cd ~/mctech/board_test
./monitor.sh                                          # observa experiments/live.csv
./monitor.sh experiments/live_modelo_260409m-2.csv     # CSV específico de 03_live_loop.py
```

Mostra CPU%, RAM, temperatura (CPU e NPU/DSP, mesmo proxy `nspss` acima) e
estado do CDSP, amostrados localmente a cada segundo, mais o FPS lido da
última linha do CSV que `03_live_loop.py` estiver escrevendo — não precisa
do runtime QAIRT carregado (só lê `/proc` e `/sys`), e não grava nenhum
arquivo. Ctrl+C encerra e restaura o cursor do terminal.

### 2. No host: `plot_results.py`

Copie os CSVs da placa pra `board_test/experiments/` (via `board_mount`/`scp`)
e rode:

```bash
cd board_test
python3 plot_results.py
```

Gera em `experiments/plots/`: um gráfico de linha por métrica (FPS, CPU%,
RAM, temp CPU, temp NPU) comparando os 3 modelos por iteração, mais um
resumo de médias. Roda numa **pyenv virtualenv própria** (não a `venv/` da
placa) — se não existir na sua máquina:
```bash
pyenv virtualenv 3.10.20 placaq6a-plots   # ou outra versão 3.10.x/3.11.x ja instalada
cd board_test
pyenv local placaq6a-plots
pip install matplotlib
```
