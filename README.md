# Ambiente de conversão de modelos — Radxa Dragon Q6A (QCS6490 / HTP v68)

Pipeline para transformar um modelo YOLOv8 treinado (`.pt`) em um modelo que a
NPU da Q6A consegue executar: `.pt → .onnx → .dlc → .dlc INT8 → context-binary`.

A **conversão e a quantização rodam aqui, no host x86** (dentro deste container).
A **inferência roda na placa** (aarch64), com o runtime QAIRT 2.42 já instalado lá.
As duas pontas usam **QAIRT 2.42** para o `.dlc`/`.bin` ser compatível.

---

## Por que um container, e por que dois ambientes Python dentro dele

O HTP do QCS6490 é integer-only: o modelo precisa ser quantizado para INT8. As
ferramentas que fazem isso (`qairt-converter`, `qairt-quantizer`,
`qnn-context-binary-generator`) vêm no QAIRT, que exige **Python 3.10 + numpy<2**.

O `ultralytics` (usado para exportar `.pt → .onnx`) puxa **numpy≥2 e protobuf
novo**, que **brigam** com o QAIRT. Se instalados juntos, um quebra o outro.

A solução é isolar em **dois ambientes virtuais** dentro do mesmo container:

- `/opt/venv-export` → ultralytics + torch + onnx (passo de export e a geração
  do dataset de calibração). Aqui o numpy≥2 é OK.
- **Python do QAIRT** (o do SDK, intocado) → passos de conversão, quantização e
  context-binary. Mantém numpy<2.

Eles **nunca compartilham numpy**. O arquivo `.onnx` no disco é a ponte entre os
dois. O `run_pipeline.sh` chama cada passo no Python certo automaticamente.

---

## Estrutura

```
q6a-conv/
├── model.env                   # fonte unica de verdade: IMGSZ/GRAPH_NAME/DSP_ARCH/SOC_ID
├── Dockerfile                  # parte da imagem oficial Radxa (QAIRT 2.42)
├── requirements-export.txt     # pacotes do venv-export (isolado)
├── run_pipeline.sh             # orquestra os passos no Python correto
├── README.md
├── scripts/
│   ├── 01_pt_to_onnx.py        # (venv-export)  .pt  -> .onnx
│   ├── 02_onnx_to_dlc.py       # (QAIRT)        .onnx -> .dlc float
│   ├── 03_quantize_dlc.py      # (QAIRT)        .dlc  -> .dlc INT8
│   └── 04_dlc_to_context.py    # (QAIRT)        .dlc INT8 -> .bin (opcional)
├── calibration/
│   ├── gen_calibration.py      # (venv-export)  dataset -> dataset de calibração
│   ├── dataset/                # SUAS imagens de calibração entram aqui
│   ├── calib_raw/              # (gerado) arquivos .raw
│   └── input_list.txt          # (gerado) lista que o quantizer consome
├── input-models/
│   └── *.pt                    # SEUS modelos treinados entram aqui
├── workspace/                  # saídas: models/modelo.onnx, .dlc, _int8.dlc, .bin
├── board_test/                 # ambiente de teste NA PLACA, em Python (webcam + qnn-net-run)
└── native_infer/                # ambiente de teste NA PLACA, em C (API QNN direta, sem CLI)
```

`input-models/`, `calibration/dataset/` e `workspace/` são montados via `-v` no
`docker run`, então você edita scripts e troca arquivos no host sem rebuildar a
imagem. `board_test/` e `native_infer/` **não** são montados no container — são
ambientes separados que rodam direto na placa (Q6A), levados até lá via `scp`
(ver seção 6 abaixo e o README de cada um).

---

## 1. Construir a imagem

> **Confirme a tag base.** O `Dockerfile` parte de
> `radxazifeng278/qairt-npu-v68:v1.2` (QAIRT 2.42). Se a Radxa publicar outra
> tag, ajuste a primeira linha do `Dockerfile`.

```bash
cd q6a-conv
sudo docker build -t q6a-conv:2.42 .
```

A imagem é grande (a base traz o SDK inteiro). O build instala só o `venv-export`
por cima.

---

## 2. Rodar o container

```bash
sudo docker run --rm -it \
  -v "$(pwd)/input-models":/workspace/input-models \
  -v "$(pwd)/scripts":/workspace/scripts \
  -v "$(pwd)/calibration":/workspace/calibration \
  -v "$(pwd)/workspace":/workspace/workspace \
  -v "$(pwd)/run_pipeline.sh":/workspace/run_pipeline.sh \
  --name q6a-conv q6a-conv:2.42 /bin/bash
```

Não precisa de `--privileged` nem `-v /dev:/dev` aqui: isto é **só conversão**, não
toca em hardware. O `--privileged` só seria necessário para inferência, que é na
placa.

### Descobrir o `envsetup.sh` do SDK (uma vez)

O `run_pipeline.sh` precisa saber onde está o `envsetup.sh` do QAIRT. Dentro do
container:

```bash
find / -name envsetup.sh 2>/dev/null
```

Se o caminho for diferente do padrão, exporte antes de rodar o pipeline:

```bash
export QAIRT_ENVSETUP=/caminho/correto/qairt/2.42.0.251225/bin/envsetup.sh
```

---

## 3. Preparar os arquivos de entrada

1. Coloque o modelo treinado em `input-models/` (ou edite `PT_PATH` no
   `scripts/01_pt_to_onnx.py` para apontar para o arquivo certo).
2. Coloque as imagens de calibração em `calibration/dataset/` (imagens reais do
   seu domínio — as fairings). ~200–500 imagens é um bom número.

---

## 4. Rodar o pipeline

Tudo de uma vez:

```bash
cd /workspace
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Ou passo a passo (útil para depurar):

```bash
./run_pipeline.sh export     # .pt  -> workspace/models/modelo.onnx
./run_pipeline.sh calib      # dataset -> calibration/calib_raw + input_list.txt
./run_pipeline.sh convert    # .onnx -> workspace/models/modelo_fp.dlc
./run_pipeline.sh quant      # -> workspace/models/modelo_int8.dlc
./run_pipeline.sh context    # -> workspace/models/modelo_int8.bin (opcional)
```

Resultado final em `workspace/`:
- `modelo_int8.dlc` — roda na placa via `qnn-net-run --dlc_path`.
- `modelo_int8.bin` — context-binary otimizado, roda via `--retrieve_context`.

---

## 5. Levar para a placa e testar

Copie o artefato para a Q6A (ajuste IP/usuário):

```bash
scp workspace/models/modelo_int8.dlc radxa@192.168.1.6:~/mctech/testePlaca/
```

Na placa (com o `env.sh` do runtime QAIRT já carregado):

```bash
source ~/mctech/testePlaca/env.sh

# usando o .dlc direto:
qnn-net-run \
  --backend "$QNN_SDK_ROOT/lib/$VARIANT/libQnnHtp.so" \
  --dlc_path modelo_int8.dlc \
  --input_list <lista_de_entradas_de_teste.txt> \
  --output_dir saida

# ou usando o context-binary:
qnn-net-run \
  --backend "$QNN_SDK_ROOT/lib/$VARIANT/libQnnHtp.so" \
  --retrieve_context modelo_int8.bin \
  --input_list <lista.txt> \
  --output_dir saida
```

As entradas de teste são `.raw` no mesmo formato da calibração (mesma resolução,
layout e normalização). Dá para reaproveitar o `gen_calibration.py` apontando
para imagens de teste.

---

## 6. Ambientes de teste mais completos na placa

Além do teste manual acima, o repo tem dois ambientes prontos, separados,
que rodam **direto na placa** (não no container) — escolha um conforme o
objetivo:

- **`board_test/`** (Python): captura frames de uma webcam USB conectada à
  placa e roda o modelo via `qnn-net-run`, headless, simulando produção.
  Mais rápido de rodar/depurar. Ver `board_test/README.md`.
- **`native_infer/`** (C): chama a API QNN diretamente (`dlopen` de
  `libQnnHtp.so`/`libQnnSystem.so`), sem `qnn-net-run` nem Python no caminho
  da inferência. Útil pra integrar em uma aplicação C/C++ de verdade, ou pra
  obter mensagens de erro mais granulares da API. Ver `native_infer/README.md`.

Os dois esperam um `.dlc`/`.bin` copiado de `workspace/models/` e o
`model.env` da raiz copiados junto via `scp` — cada README tem o comando
exato.

---

## Pontos de atenção (onde as coisas costumam quebrar)

- **Versão casada (host ↔ placa):** ambos em QAIRT **2.42**. Um `.dlc` roda na
  versão que o gerou e em versões mais novas — nunca em mais antigas. Não misture.

- **Pré-processamento idêntico:** o `gen_calibration.py` precisa reproduzir
  exatamente a resolução, o layout (NCHW), a ordem de canais (RGB) e a
  normalização (`/255`) que o modelo espera e que a placa vai usar. Se divergir,
  a quantização calibra errado e a acurácia cai.

- **Nome/shape do input no passo 02:** `INPUT_NAME`/`INPUT_SHAPE` precisam casar
  com o ONNX. Confira o nome real do tensor (ex.: com netron) — para YOLOv8
  costuma ser `images`, shape `1,3,640,640`.

- **`GRAPH_NAME` não é um nome livre:** o `qairt-converter` nomeia o grafo
  dentro do `.dlc`/`.bin` a partir do *nome do arquivo* de `DLC_OUT` (passo
  02), não de um valor de config separado. Se renomear `DLC_OUT`, atualize
  `GRAPH_NAME` em `model.env` também — o `native_infer/qnn_infer` lista os
  grafos reais do `.bin` se o nome configurado não bater, o que ajuda a
  descobrir o valor certo.

- **Flags do quantizer:** esta versão usa `--act_quantizer_calibration` /
  `--param_quantizer_calibration`. A flag antiga `--quant_scheme` **não existe
  mais** no 2.42. Se editar o script, rode `qairt-quantizer --help` para
  confirmar os nomes na sua versão.

- **opset do ONNX:** se o `qairt-converter` reclamar de operador não suportado,
  ajuste `OPSET` no passo 01 (tente 11 ou 13).

- **soc_id / dsp_arch (passo 04):** `v68` e `soc_id 35` são da Q6A (QCS6490).
  Não altere a menos que troque de placa.

---

## Valores padrão x personalização

Todos os scripts têm um bloco `CONFIG` no topo com valores padrão e comentários
ao lado explicando o que alterar e quando. Não há `argparse` de propósito: a
entrada é o bloco `CONFIG`, para o fluxo ficar explícito e reproduzível. Edite o
bloco conforme sua necessidade (caminhos, resolução, opset, método de
calibração, bitwidths).

Os valores que precisam ficar **idênticos entre a conversão e os dois
ambientes de placa** (`IMGSZ`, `INPUT_NAME`, `GRAPH_NAME`, `DSP_ARCH`,
`SOC_ID`) não são mais duplicados em cada script — todos leem o default de
`model.env` (raiz do repo), que é a fonte única de verdade. Mude o modelo ou a
resolução? Edite só o `model.env`. Cada script continua funcionando sozinho
mesmo sem esse arquivo (cai num valor hardcoded de fallback), mas aí corre o
risco de divergir do que foi usado pra gerar o modelo — então mantenha o
`model.env` atualizado e copie-o pra placa de novo depois de qualquer
mudança (ver seção 6).
