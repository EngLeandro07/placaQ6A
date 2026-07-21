# Inferência nativa em C (API QNN direta, sem CLI)

Cliente C puro que chama a API QNN diretamente (`libQnnHtp.so`/`libQnnSystem.so`
via `dlopen`), sem passar pelo `qnn-net-run` (CLI) nem por wrapper Python. Roda
o mesmo `.bin` (context-binary) gerado pelo passo 04 do pipeline principal.

Este workspace é separado do `board_test/` (que continua sendo a captura de
webcam em Python + `qnn-net-run` via CLI). Aqui o foco é só a inferência,
via API nativa — a captura de frames continua em Python (`board_test/`).

## Sobre o bug de DMA/FastRPC da placa (e o workaround)

O crash de "no reserved DMA memory for FASTRPC" / segfault no CDSP é um
problema de kernel/firmware da imagem Radxa R2 (ver memória do projeto /
tópico aberto no fórum da Radxa), não do jeito que a inferência é chamada —
o `native_infer` bate no mesmo erro que o `qnn-net-run` bateria, confirmando
que não é bug de nenhuma ferramenta específica.

**Existe workaround**: testado empiricamente (2026-07-20) que o device tree
desta placa só tem ~2MB de memória DMA reservada utilizável pro FastRPC/CDSP,
não os 4-8MB que o QAIRT pede por padrão. Gerando o `.bin` no passo 04 do
pipeline com `VTCM_MB<=2` (já é o default em `conversion/scripts/
04_dlc_to_context.py`), a inferência roda de verdade na NPU — mais lenta que
o ideal (mais *spill* pra DDR), mas funcional. Se `native_infer` ainda bater
nesse erro, confira se o `.bin` que você copiou pra placa foi gerado com esse
`VTCM_MB` baixo (reconverta se necessário). A API QNN native pode dar
mensagens de erro mais detalhadas que o `qnn-net-run` (via
`QnnError_getMessage`/`getVerboseMessage`, não usado ainda nesta primeira
versão) — útil se quiser investigar mais o comportamento exato do erro.

## Levar para a placa

**Opção 1 — `board/` montado (recomendado, ver `board_mount.sh` na raiz):**
```bash
./board_mount.sh mount        # uma vez por sessão, monta ~/mctech em ./board/
cp -r native_infer model.env board/
mkdir -p board/models   # ~/mctech/models/ - COMPARTILHADO com board_test/, não fica dentro de native_infer/
cp conversion/output-models/modelo_int8.bin board/models/
```
`board/` é o mesmo filesystem da placa (via `sshfs`) — os `cp` já escrevem
direto lá, sem `scp` repetido. `models/` fica um nível acima de
`native_infer/` (irmão dele e de `board_test/`, mesmo espírito de
`qairt_runtime/`/`model.env`) porque os dois ambientes usam os MESMOS
artefatos `.dlc`/`.bin` — evita duplicar a mesma cópia em cada um.

**Opção 2 — `scp` direto (se não tiver o mount configurado):**
```bash
scp -r native_infer radxa@<ip-da-placa>:~/mctech/native_infer
scp model.env radxa@<ip-da-placa>:~/mctech/native_infer/
ssh radxa@<ip-da-placa> mkdir -p ~/mctech/models
scp conversion/output-models/modelo_int8.bin radxa@<ip-da-placa>:~/mctech/models/
```

## Compilar (na placa)

```bash
export QAIRT_SDK_ROOT=$HOME/mctech/qairt_runtime/qairt/2.42.0.251225
cd ~/mctech/native_infer
make
```

O `Makefile` lê `GRAPH_NAME` de `model.env` (procura primeiro ao lado dele
mesmo, depois em `../model.env`) e compila esse valor como default no
binário via `-DDEFAULT_GRAPH_NAME`. Sem `model.env`, cai no fallback
`modelo_fp` (mesmo default hardcoded no `.c`).

## Rodar (na placa, com o runtime QAIRT sourced)

Dois modos, mesmo binário:

```bash
source ~/mctech/qairt_runtime/env.sh   # deixa libQnnHtp.so/libQnnSystem.so no LD_LIBRARY_PATH

# modo single-shot (original): 1 frame por invocação, processo sobe e morre
./qnn_infer ../models/modelo_int8.bin frame.raw
# ou, pra sobrescrever o nome do grafo default (compilado a partir de model.env):
# ./qnn_infer ../models/modelo_int8.bin frame.raw outro_nome_de_grafo

# modo loop (novo, ver seção abaixo): carrega o modelo 1x, executa N frames
./qnn_infer --loop ../models/modelo_int8.bin
```

- `../models/modelo_int8.bin`: context-binary gerado pelo passo 04 (`conversion/output-models/modelo_int8.bin`
  no repo do host, copiado via `scp`/`cp` pra dentro de `~/mctech/models/` — diretório
  **compartilhado** com `board_test/`, um nível acima de `native_infer/`, não dentro dele.
- `frame.raw` (só no modo single-shot): um frame já pré-processado, em
  **uint8 bruto (0-255)**, NCHW, RGB, **sem normalização**
  (`board_test/captures/*.raw` já gera nesse formato desde 2026-07-20). O
  grafo INT8 espera pixel bruto - a normalização fica embutida na própria
  quantização do grafo. Descoberto testando este programa: o tensor de
  entrada rejeitava um `.raw` float32 normalizado por ter 4x o tamanho
  esperado.
- nome do grafo (último argumento, opcional nos dois modos): **não é
  livre** — o `qairt-converter` usa o *stem* do `DLC_OUT` do passo 02 como
  nome interno do grafo, **sanitizado como identificador C** (hífen vira
  `_`, e se começar com dígito ganha um `_` na frente - ex.:
  `260417_1280_nano_fp` no `DLC_OUT` vira o grafo `_260417_1280_nano_fp`
  dentro do `.bin`). Ver comentário em `model.env`, chave `GRAPH_NAME`. Se
  você não tiver certeza do valor certo, rode sem esse argumento: o
  programa lista os grafos encontrados no `.bin` (em stderr) e usa o
  default compilado; se não bater, ele avisa e você roda de novo passando o
  nome certo (já sanitizado, como aparece na listagem).

### Modo loop (`--loop`)

Pensado pra ser chamado como **subprocess persistente** por um
orquestrador externo — hoje isso é `board_test/03_live_loop.py`, que sobe
`qnn_infer --loop` uma vez no início e manda um frame por vez enquanto a
webcam captura continuamente. Faz todo o setup (dlopen, backend, device,
`contextCreateFromBinary`, `graphRetrieve`) **uma única vez**, depois fica
lendo frames do stdin em loop, um `graphExecute` por frame, até o stdin
fechar (EOF) — é isso que faltava pra medir desempenho real de uso
contínuo (câmera liga uma vez, inferência roda em loop); antes disso, um
benchmark de FPS contra este binário media sobretudo o overhead de reload
repetido (ver "Sobre benchmark comparativo" abaixo).

```bash
./qnn_infer --loop <modelo.bin> [nome_do_grafo]
```

Protocolo no **stdin** (por frame, síncrono — um round-trip por frame, sem
pipelining):
```
4 bytes little-endian (uint32) = N, tamanho do frame em bytes
N bytes do frame (mesmo layout NCHW/uint8 do modo single-shot)
```

Resposta no **stdout** (por frame), uma linha de texto ASCII:
```
OK <exec_us> <min> <max> <mean>\n   # sucesso - stats do 1º tensor de saída
ERR tamanho_invalido\n              # frame não bate com o tensor de entrada
```
`ERR` **não é fatal** — o processo continua vivo e aguarda o próximo frame
(o custo de recarregar o `.bin` é caro demais pra matar o processo inteiro
por causa de 1 frame ruim). EOF limpo no stdin (nenhum byte do próximo
header) é desligamento normal, com `exit(0)` depois do teardown; EOF no
**meio** de um header ou payload é tratado como stream corrompido — aí sim
é fatal, porque não dá mais pra confiar no alinhamento do framing.

No modo loop, **todo diagnóstico** (setup, stats completas de todas as
saídas por frame) vai para **stderr** — stdout fica reservado
exclusivamente pro protocolo `OK`/`ERR` linha-a-linha acima, pra um
orquestrador como `03_live_loop.py` poder consumir com um simples
`readline()`. No modo single-shot isso não muda (stdout continua sendo o
resultado que um humano rodando manualmente quer ver na tela) — só o ruído
de setup migrou pra stderr nos dois modos.

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

## Bugs corrigidos (2026-07-20)

Nenhum dos dois bugs abaixo tinha aparecido antes porque `contextCreateFromBinary`
sempre falhava primeiro por causa do bug de VTCM/DMA (ver seção acima) — só
depois do workaround (`VTCM_MB<=2`) o programa chegou longe o suficiente pra
exercitar esse código pela primeira vez:

- **Use-after-free**: `systemContextFree(sysCtx)` rodava logo depois de
  `graphRetrieve`, mas `inputDescs`/`outputDescs` (e os tensores de execução
  montados a partir deles, com `name`/`dimensions` apontando pra dentro da
  memória do system-context) só são usados DEPOIS disso, em `buildExecTensor`
  e `graphExecute`. Causava segfault. Corrigido movendo o `systemContextFree`
  pra depois do `graphExecute` (e do print de estatísticas, que também lê
  `name`).
- **ID de tensor hardcoded em 0**: `buildExecTensor` setava `t.v1.id = 0`
  pra todo tensor, mas `graphExecute` casa os tensores fornecidos pelo ID
  real que o grafo espera (ex.: `images`=1, `output0`=1026) — com id=0
  sempre errado, falhava com "Expected Tensor ID: N not found in
  user-provided tensors". Corrigido lendo o ID de verdade do `desc`
  (`TENSOR_GET_ID`).

## Sobre benchmark comparativo (FPS/CPU/RAM/temperatura)

**Resolvido**: `qnn_infer` ganhou o modo `--loop` (ver seção acima), que
carrega o modelo uma única vez e executa múltiplos frames em sequência —
exatamente o que faltava pra medir desempenho real de uso contínuo (câmera
liga uma vez, inferência roda em loop), em vez do overhead de reload
repetido que um benchmark contra o modo single-shot mediria. A forma válida
de medir FPS/CPU/RAM/temperatura do `native_infer` em produção hoje é rodar
`board_test/03_live_loop.py`, que sobe `qnn_infer --loop` como subprocess
persistente e alimenta frame a frame da webcam, gravando um CSV
(`experiments/live_<nome>.csv`) — acompanhe ao vivo com `board_test/monitor.sh`
em outro terminal.

`board_test/benchmark_batch.sh` (antigo `monitor.sh`) continua cobrindo só
o fluxo via `qnn-net-run` (`board_test/`, N iterações sobre um lote fixo) —
esse formato específico de "N iterações sobre o mesmo lote" não faz sentido
pro `native_infer`, já que o modo `--loop` foi desenhado pra uso contínuo
real (frame a frame, ao vivo), não pra repetir o mesmo lote em memória.

## Limitações desta primeira versão

- Assume 1 único tensor de entrada (`inputs[0]`) — suficiente pro YOLO deste
  projeto (imagem única, NCHW).
- Não decodifica as detecções do YOLO (NMS/grid/anchors) — só confirma que a
  inferência roda e que a saída não é lixo (min/max/mean plausíveis).
- Não usa `QnnError_getMessage`/`getVerboseMessage` para mensagens de erro
  detalhadas ainda — só o código de erro numérico. Vale adicionar se
  precisarmos diagnosticar melhor o erro de VTCM/DMA por este caminho.
