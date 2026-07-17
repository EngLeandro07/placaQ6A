# Inferência nativa em C (API QNN direta, sem CLI)

Cliente C puro que chama a API QNN diretamente (`libQnnHtp.so`/`libQnnSystem.so`
via `dlopen`), sem passar pelo `qnn-net-run` (CLI) nem por wrapper Python. Roda
o mesmo `.bin` (context-binary) gerado pelo passo 04 do pipeline principal.

Este workspace é separado do `board_test/` (que continua sendo a captura de
webcam em Python + `qnn-net-run` via CLI). Aqui o foco é só a inferência,
via API nativa — a captura de frames continua em Python (`board_test/`).

## Por que isso não conserta o crash que já vimos

O crash de "no reserved DMA memory for FASTRPC" / segfault no CDSP é um
problema de kernel/firmware da placa (ver memória do projeto / tópico aberto
no fórum da Radxa), não do jeito que a inferência é chamada. Este programa
provavelmente vai bater no mesmo erro até isso ser resolvido — a vantagem é
que a API QNN native pode dar mensagens de erro mais detalhadas que o
`qnn-net-run` (via `QnnError_getMessage`/`getVerboseMessage`, não usado ainda
nesta primeira versão).

## Levar para a placa

Do host (dentro deste repo):
```bash
scp -r native_infer radxa@<ip-da-placa>:~/mctech/native_infer
scp model.env radxa@<ip-da-placa>:~/mctech/native_infer/
scp workspace/models/modelo_int8.bin radxa@<ip-da-placa>:~/mctech/native_infer/
```

## Compilar (na placa)

```bash
export QAIRT_SDK_ROOT=$HOME/mctech/testePlaca/qairt/2.42.0.251225
cd ~/mctech/native_infer
make
```

O `Makefile` lê `GRAPH_NAME` de `model.env` (procura primeiro ao lado dele
mesmo, depois em `../model.env`) e compila esse valor como default no
binário via `-DDEFAULT_GRAPH_NAME`. Sem `model.env`, cai no fallback
`modelo_fp` (mesmo default hardcoded no `.c`).

## Rodar (na placa, com o runtime QAIRT sourced)

```bash
source ~/mctech/testePlaca/env.sh   # deixa libQnnHtp.so/libQnnSystem.so no LD_LIBRARY_PATH
./qnn_infer modelo_int8.bin frame.raw
# ou, pra sobrescrever o nome do grafo default (compilado a partir de model.env):
# ./qnn_infer modelo_int8.bin frame.raw outro_nome_de_grafo
```

- `modelo_int8.bin`: context-binary gerado pelo passo 04 (`workspace/models/modelo_int8.bin`
  no repo do host, copiado via `scp`).
- `frame.raw`: um frame já pré-processado (mesmo formato da calibração/
  `board_test/captures/*.raw` — NCHW, RGB, float32, `/255`).
- nome do grafo (3º argumento, opcional): **não é livre** — o
  `qairt-converter` usa o *stem* do `DLC_OUT` do passo 02 como nome interno
  do grafo (ver comentário em `model.env`, chave `GRAPH_NAME`). Se você não
  tiver certeza do valor certo, rode sem o 3º argumento: o programa lista os
  grafos encontrados no `.bin` e usa o default compilado; se não bater, ele
  avisa e você roda de novo passando o nome certo.

## O que o programa faz

1. Carrega `libQnnSystem.so` e `libQnnHtp.so` via `dlopen`, resolve os
   ponteiros de função da API (`QnnInterface_getProviders`).
2. `logCreate` → `backendCreate` → `deviceCreate`.
3. Lê o `.bin` inteiro pra memória e usa `QnnSystemContext_getBinaryInfo`
   pra descobrir os grafos e o shape/dtype dos tensores de entrada/saída —
   não precisa saber de antemão a arquitetura do modelo.
4. `contextCreateFromBinary` (carrega o `.bin` na HTP) → `graphRetrieve`.
5. Monta os tensores de execução (aloca buffers do tamanho certo), carrega o
   `.raw` de entrada, roda `graphExecute`.
6. Imprime estatísticas básicas (min/max/mean) de cada saída — sem decodificar
   as detecções do YOLO (mesmo espírito do `board_test/02_run_inference.py`).
7. Libera tudo (`contextFree`/`deviceFree`/`backendFree`/`logFree`/`dlclose`).

## Limitações desta primeira versão

- Assume 1 único tensor de entrada (`inputs[0]`) — suficiente pro YOLO deste
  projeto (imagem única, NCHW).
- Não decodifica as detecções do YOLO (NMS/grid/anchors) — só confirma que a
  inferência roda e que a saída não é lixo (min/max/mean plausíveis).
- Não usa `QnnError_getMessage`/`getVerboseMessage` para mensagens de erro
  detalhadas ainda — só o código de erro numérico. Vale adicionar se
  precisarmos diagnosticar melhor o erro de VTCM/DMA por este caminho.
