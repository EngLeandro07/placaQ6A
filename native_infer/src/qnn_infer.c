/* =============================================================================
 *  qnn_infer.c
 *  Cliente NATIVO em C puro para a API QNN (Qualcomm) - roda o context-binary
 *  (.bin) gerado pelo passo 04 direto na HTP, sem passar pelo qnn-net-run
 *  (CLI) nem por wrapper Python/subprocess. So' a API QNN, chamada direto.
 *
 *  ONDE RODA: na PLACA, com o runtime QAIRT carregado (source env.sh), pra
 *  que libQnnHtp.so/libQnnSystem.so estejam no LD_LIBRARY_PATH.
 *
 *  DOIS MODOS:
 *
 *  1) single-shot (uso original): 1 frame por invocacao, processo inteiro
 *     sobe e morre a cada chamada (setup completo + graphExecute + saida).
 *        ./qnn_infer <modelo.bin> <input.raw> [nome_do_grafo]
 *
 *  2) --loop (setup 1x, executa N frames): pensado pra ser chamado como
 *     subprocess persistente por um orquestrador externo (ex.
 *     board_test/03_live_loop.py) - carrega o .bin e cria o contexto UMA VEZ,
 *     depois fica lendo frames do stdin em loop, um graphExecute por frame,
 *     ate' o stdin fechar (EOF). Isso e' o que faltava pra medir desempenho
 *     real de uso continuo (camera liga uma vez, inferencia roda em loop) -
 *     antes disso, um benchmark de FPS contra este binario media sobretudo
 *     overhead de reload repetido (ver native_infer/README.md).
 *        ./qnn_infer --loop <modelo.bin> [nome_do_grafo]
 *     Protocolo no stdin (por frame, sincrono - um round-trip por frame):
 *        4 bytes little-endian (uint32) = N, tamanho do frame em bytes
 *        N bytes do frame (mesmo layout NCHW/uint8 do modo single-shot)
 *     Resposta no stdout (por frame):
 *        "OK <exec_us> <min> <max> <mean>\n"  (stats do 1o tensor de saida)
 *        ou "ERR tamanho_invalido\n" se o frame nao bater com o tensor de
 *        entrada esperado - NAO e' fatal, o processo continua vivo (o custo
 *        de recarregar o .bin e' caro demais pra matar o processo por causa
 *        de 1 frame ruim). EOF limpo no stdin (nenhum byte do proximo header)
 *        e' desligamento normal; EOF no MEIO de um header/payload e' stream
 *        corrompido, ai sim fatal (nao da' mais pra confiar no alinhamento
 *        do framing). No modo loop, TODO diagnostico (setup, stats completas
 *        de todas as saidas por frame) vai pra stderr - stdout fica reservado
 *        exclusivamente pro protocolo OK/ERR linha-a-linha acima.
 *
 *  SEQUENCIA DE SETUP (extraida do QnnSampleApp.cpp oficial do SDK, mesma
 *  ordem, roda 1x em qualquer um dos dois modos - ver qnnSetup() abaixo):
 *    dlopen(libQnnSystem.so) -> QnnSystemInterface_getProviders
 *    dlopen(libQnnHtp.so)    -> QnnInterface_getProviders
 *    logCreate -> backendCreate -> deviceCreate
 *    systemContextCreate -> systemContextGetBinaryInfo (le' metadados do .bin:
 *      nomes/shapes/dtypes dos grafos e tensores, sem precisar saber de
 *      antemao o formato do modelo)
 *    contextCreateFromBinary (carrega o .bin na HTP)
 *    graphRetrieve (pega o grafo pelo nome)
 *  Por frame (qnnRunFrame(), 1x no single-shot, N vezes no loop):
 *    graphExecute
 *  No teardown (qnnTeardown(), 1x, so' apos o ULTIMO graphExecute):
 *    systemContextFree -> contextFree -> deviceFree -> backendFree -> logFree
 *
 *  NAO conserta o crash de DMA/VTCM ja diagnosticado no CDSP da placa - so'
 *  troca COMO a inferencia e' chamada (nativo em vez de CLI, com ou sem
 *  reload por frame). Se o problema de firmware/FastRPC nao estiver
 *  resolvido, isto vai falhar do mesmo jeito (mas com mensagens de erro
 *  potencialmente mais detalhadas via QnnError).
 *
 *  Compilar na propria placa (ver Makefile / README.md).
 * =============================================================================
 */

/* Necessario pra clock_gettime/CLOCK_MONOTONIC/struct timespec ficarem
 * visiveis com -std=c11 (sem isso, o glibc esconde essas declaracoes atras
 * de __STRICT_ANSI__ por padrao). Tem que vir ANTES de qualquer #include. */
#define _POSIX_C_SOURCE 199309L

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

/* =============================== CONFIG ==================================== */
/* Nome das libs do backend HTP e do system-lib. Resolvidos via LD_LIBRARY_PATH
 * (setado pelo env.sh do runtime QAIRT) - nao precisam de caminho absoluto. */
#define BACKEND_LIB_NAME "libQnnHtp.so"
#define SYSTEM_LIB_NAME  "libQnnSystem.so"

/* Nome do grafo default, se nao passado por argv. O Makefile le' GRAPH_NAME
 * de model.env (raiz do repo, fonte unica de verdade) e passa via -D na
 * compilacao; se nao passado (compilado fora do Makefile), cai neste
 * fallback. Ver model.env pra entender de onde esse nome vem. */
#ifndef DEFAULT_GRAPH_NAME
#define DEFAULT_GRAPH_NAME "modelo_fp"
#endif

/* Teto de sanidade pro tamanho de 1 frame no protocolo do modo --loop. Um N
 * acima disso no header e' tratado como stream corrompido (fatal), nao como
 * um frame legitimo grande - generoso o bastante pra qualquer resolucao
 * razoavel deste projeto (1280x1280x3 uint8 ~= 4.9MB). */
#define MAX_LOOP_FRAME_BYTES (64u * 1024u * 1024u)
/* ============================================================================ */

static void logCallback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
  (void)level;
  (void)timestamp;
  vfprintf(stderr, fmt, args);
}

static void *mustDlopen(const char *name) {
  void *h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "[erro] dlopen(%s) falhou: %s\n", name, dlerror());
    exit(1);
  }
  return h;
}

static void *mustDlsym(void *handle, const char *name) {
  void *sym = dlsym(handle, name);
  if (!sym) {
    fprintf(stderr, "[erro] dlsym(%s) falhou: %s\n", name, dlerror());
    exit(1);
  }
  return sym;
}

static void checkQnn(Qnn_ErrorHandle_t status, const char *what) {
  if (status != QNN_SUCCESS) {
    fprintf(stderr, "[erro] %s falhou, codigo QNN: 0x%llx\n", what,
            (unsigned long long)status);
    exit(1);
  }
}

/* Le' um arquivo inteiro para um buffer malloc'd. Preenche *outSize. */
static void *readFile(const char *path, long *outSize) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[erro] nao consegui abrir %s: %s\n", path, strerror(errno));
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *buf = malloc((size_t)size);
  if (!buf) {
    fprintf(stderr, "[erro] malloc(%ld) falhou lendo %s\n", size, path);
    exit(1);
  }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    fprintf(stderr, "[erro] fread incompleto em %s\n", path);
    exit(1);
  }
  fclose(f);
  *outSize = size;
  return buf;
}

/* Le' exatamente n bytes de stdin pra buf. Retorna quantos bytes foram
 * efetivamente lidos - < n so' acontece em EOF (produtor fechou o pipe). */
static size_t readExactStdin(uint8_t *buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    size_t r = fread(buf + got, 1, n - got, stdin);
    if (r == 0) break;
    got += r;
  }
  return got;
}

/* ---- Helpers de dispatch de versao do Qnn_Tensor_t (V1 ou V2) --------------
 * O .bin pode descrever tensores em V1 ou V2 (QnnSystemContext_GraphInfoV*).
 * So' precisamos de um subconjunto pequeno de campos (id/name/dataType/
 * rank/dimensions/memType/clientBuf) - os dois tem os mesmos primeiros
 * campos com o mesmo layout, entao os macros abaixo dao conta dos dois. */
#define TENSOR_GET_NAME(t)      (((t).version == QNN_TENSOR_VERSION_1) ? (t).v1.name : (t).v2.name)
#define TENSOR_GET_ID(t)        (((t).version == QNN_TENSOR_VERSION_1) ? (t).v1.id : (t).v2.id)
#define TENSOR_GET_DATATYPE(t)  (((t).version == QNN_TENSOR_VERSION_1) ? (t).v1.dataType : (t).v2.dataType)
#define TENSOR_GET_RANK(t)      (((t).version == QNN_TENSOR_VERSION_1) ? (t).v1.rank : (t).v2.rank)
#define TENSOR_GET_DIMS(t)      (((t).version == QNN_TENSOR_VERSION_1) ? (t).v1.dimensions : (t).v2.dimensions)

/* Tamanho em bytes de 1 elemento, a partir do encoding QNN_DATATYPE_xxx
 * (byte baixo do enum = largura em bits; ver comentario em QnnTypes.h). */
static size_t dtypeElemSize(Qnn_DataType_t dt) {
  return (size_t)(dt & 0xFF) / 8;
}

/* Monta um Qnn_Tensor_t "de execucao" (memType RAW, buffer malloc'd) a partir
 * da descricao vinda do system-context (nome/shape/dtype), alocando o buffer
 * do tamanho certo. rawDataOut recebe o ponteiro do buffer pra quem chamou
 * poder ler/escrever os dados brutos. */
static Qnn_Tensor_t buildExecTensor(const Qnn_Tensor_t *desc, uint8_t **rawDataOut) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;

  uint32_t rank = TENSOR_GET_RANK(*desc);
  uint32_t *dims = TENSOR_GET_DIMS(*desc);
  Qnn_DataType_t dtype = TENSOR_GET_DATATYPE(*desc);

  size_t numElements = 1;
  for (uint32_t i = 0; i < rank; i++) numElements *= dims[i];
  size_t numBytes = numElements * dtypeElemSize(dtype);

  uint8_t *raw = (uint8_t *)malloc(numBytes);
  if (!raw) {
    fprintf(stderr, "[erro] malloc(%zu) falhou pro tensor %s\n", numBytes,
            TENSOR_GET_NAME(*desc));
    exit(1);
  }
  memset(raw, 0, numBytes);

  /* o ID precisa bater com o que o grafo espera (graphExecute rejeita com
   * "Expected Tensor ID: N not found in user-provided tensors" se nao
   * bater) - NAO e' um valor livre, tem que vir do proprio desc. */
  t.v1.id = TENSOR_GET_ID(*desc);
  t.v1.name = TENSOR_GET_NAME(*desc);
  t.v1.type = QNN_TENSOR_TYPE_APP_READWRITE;
  t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = dtype;
  t.v1.rank = rank;
  t.v1.dimensions = dims;
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf.data = raw;
  t.v1.clientBuf.dataSize = (uint32_t)numBytes;

  *rawDataOut = raw;
  return t;
}

/* ---- Contexto persistente entre frames (setup feito 1x) -------------------- */
typedef struct {
  void *systemLibHandle, *backendLibHandle;
  const QNN_INTERFACE_VER_TYPE *qnn;
  const QNN_SYSTEM_INTERFACE_VER_TYPE *qnnSystem;
  Qnn_LogHandle_t logHandle;
  Qnn_BackendHandle_t backendHandle;
  Qnn_DeviceHandle_t deviceHandle;
  Qnn_ContextHandle_t context;
  Qnn_GraphHandle_t graphHandle;
  QnnSystemContext_Handle_t sysCtx; /* liberado SO' no teardown, ver nota em qnnTeardown() */
  void *binBuffer;
  uint32_t numInputs, numOutputs;
  Qnn_Tensor_t *inputs, *outputs;
  uint8_t **inputBufs, **outputBufs;
} QnnCtx;

/* Faz todo o setup que so' precisa acontecer 1x, seja no modo single-shot
 * (1 frame) ou --loop (N frames): dlopen das libs, backend/device, leitura
 * do .bin, resolucao do grafo pelo nome, criacao do contexto na HTP, e
 * alocacao dos tensores de execucao (buffers reutilizados em todo frame
 * subsequente, mesmo tamanho sempre - shape fixo do grafo). Sai do processo
 * (exit(1)) em qualquer falha, igual ao comportamento de sempre. */
static void qnnSetup(QnnCtx *ctx, const char *binPath, const char *graphName) {
  memset(ctx, 0, sizeof(*ctx));

  /* ---- 1. Carrega libQnnSystem.so e libQnnHtp.so, resolve os providers ---- */
  ctx->systemLibHandle = mustDlopen(SYSTEM_LIB_NAME);
  ctx->backendLibHandle = mustDlopen(BACKEND_LIB_NAME);

  typedef Qnn_ErrorHandle_t (*GetProvidersFn_t)(const QnnInterface_t ***, uint32_t *);
  typedef Qnn_ErrorHandle_t (*GetSystemProvidersFn_t)(const QnnSystemInterface_t ***, uint32_t *);

  GetProvidersFn_t getProviders =
      (GetProvidersFn_t)mustDlsym(ctx->backendLibHandle, "QnnInterface_getProviders");
  GetSystemProvidersFn_t getSystemProviders =
      (GetSystemProvidersFn_t)mustDlsym(ctx->systemLibHandle, "QnnSystemInterface_getProviders");

  const QnnInterface_t **providers = NULL;
  uint32_t numProviders = 0;
  checkQnn(getProviders(&providers, &numProviders), "QnnInterface_getProviders");

  ctx->qnn = NULL;
  for (uint32_t i = 0; i < numProviders; i++) {
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
      ctx->qnn = &providers[i]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!ctx->qnn) {
    fprintf(stderr, "[erro] nenhum provider QNN compativel (API major %d)\n",
            QNN_API_VERSION_MAJOR);
    exit(1);
  }

  const QnnSystemInterface_t **sysProviders = NULL;
  uint32_t numSysProviders = 0;
  checkQnn(getSystemProviders(&sysProviders, &numSysProviders),
           "QnnSystemInterface_getProviders");
  ctx->qnnSystem = NULL;
  for (uint32_t i = 0; i < numSysProviders; i++) {
    if (sysProviders[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR) {
      ctx->qnnSystem = &sysProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!ctx->qnnSystem) {
    fprintf(stderr, "[erro] nenhum system-provider compativel\n");
    exit(1);
  }

  /* ---- 2. log / backend / device -------------------------------------- */
  checkQnn(ctx->qnn->logCreate(logCallback, QNN_LOG_LEVEL_WARN, &ctx->logHandle), "logCreate");
  checkQnn(ctx->qnn->backendCreate(ctx->logHandle, NULL, &ctx->backendHandle), "backendCreate");
  checkQnn(ctx->qnn->deviceCreate(ctx->logHandle, NULL, &ctx->deviceHandle), "deviceCreate");

  fprintf(stderr, "[setup] backend/device criados. carregando %s...\n", binPath);

  /* ---- 3. le' o .bin e extrai metadados via system-context ------------- */
  long binSize = 0;
  ctx->binBuffer = readFile(binPath, &binSize);

  checkQnn(ctx->qnnSystem->systemContextCreate(&ctx->sysCtx), "systemContextCreate");

  const QnnSystemContext_BinaryInfo_t *binaryInfo = NULL;
  Qnn_ContextBinarySize_t binaryInfoSize = 0;
  checkQnn(ctx->qnnSystem->systemContextGetBinaryInfo(ctx->sysCtx, ctx->binBuffer,
                                                       (uint64_t)binSize, &binaryInfo,
                                                       &binaryInfoSize),
           "systemContextGetBinaryInfo");

  uint32_t numGraphs = 0;
  QnnSystemContext_GraphInfo_t *graphs = NULL;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
    graphs = binaryInfo->contextBinaryInfoV1.graphs;
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
    graphs = binaryInfo->contextBinaryInfoV2.graphs;
  } else {
    numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
    graphs = binaryInfo->contextBinaryInfoV3.graphs;
  }
  fprintf(stderr, "[setup] %s tem %u grafo(s)\n", binPath, numGraphs);

  /* acha o grafo pelo nome. NAO copiamos os Qnn_Tensor_t - so' guardamos
   * ponteiros (inputDescs/outputDescs) pra dentro da memoria alocada pelo
   * proprio system-context (nao pra dentro de binBuffer). Por isso
   * systemContextFree() SO' PODE rodar depois do ultimo uso desses
   * ponteiros (em qualquer graphExecute/print de stats, no processo
   * inteiro) - liberar antes e' use-after-free (ver nota em qnnTeardown). */
  QnnSystemContext_GraphInfo_t *targetGraph = NULL;
  for (uint32_t i = 0; i < numGraphs; i++) {
    const char *name = (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
                            ? graphs[i].graphInfoV1.graphName
                        : (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2)
                            ? graphs[i].graphInfoV2.graphName
                            : graphs[i].graphInfoV3.graphName;
    fprintf(stderr, "[setup]   grafo[%u]: %s\n", i, name);
    if (strcmp(name, graphName) == 0) targetGraph = &graphs[i];
  }
  if (!targetGraph) {
    fprintf(stderr, "[erro] grafo '%s' nao encontrado no .bin\n", graphName);
    exit(1);
  }

  uint32_t numInputs = 0, numOutputs = 0;
  Qnn_Tensor_t *inputDescs = NULL, *outputDescs = NULL;
  if (targetGraph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    numInputs = targetGraph->graphInfoV1.numGraphInputs;
    inputDescs = targetGraph->graphInfoV1.graphInputs;
    numOutputs = targetGraph->graphInfoV1.numGraphOutputs;
    outputDescs = targetGraph->graphInfoV1.graphOutputs;
  } else if (targetGraph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    numInputs = targetGraph->graphInfoV2.numGraphInputs;
    inputDescs = targetGraph->graphInfoV2.graphInputs;
    numOutputs = targetGraph->graphInfoV2.numGraphOutputs;
    outputDescs = targetGraph->graphInfoV2.graphOutputs;
  } else {
    numInputs = targetGraph->graphInfoV3.numGraphInputs;
    inputDescs = targetGraph->graphInfoV3.graphInputs;
    numOutputs = targetGraph->graphInfoV3.numGraphOutputs;
    outputDescs = targetGraph->graphInfoV3.graphOutputs;
  }
  fprintf(stderr, "[setup] grafo '%s': %u entrada(s), %u saida(s)\n", graphName, numInputs, numOutputs);

  /* ---- 4. cria o contexto a partir do binario (carrega na HTP) --------- */
  checkQnn(ctx->qnn->contextCreateFromBinary(ctx->backendHandle, ctx->deviceHandle, NULL,
                                              ctx->binBuffer, (Qnn_ContextBinarySize_t)binSize,
                                              &ctx->context, NULL),
           "contextCreateFromBinary");
  checkQnn(ctx->qnn->graphRetrieve(ctx->context, graphName, &ctx->graphHandle), "graphRetrieve");

  fprintf(stderr, "[setup] contexto carregado na HTP, grafo '%s' recuperado.\n", graphName);

  /* ---- 5. monta tensores de execucao (buffers reaproveitados em todo frame) */
  ctx->numInputs = numInputs;
  ctx->numOutputs = numOutputs;
  ctx->inputs = (Qnn_Tensor_t *)calloc(numInputs, sizeof(Qnn_Tensor_t));
  ctx->outputs = (Qnn_Tensor_t *)calloc(numOutputs, sizeof(Qnn_Tensor_t));
  ctx->inputBufs = (uint8_t **)calloc(numInputs, sizeof(uint8_t *));
  ctx->outputBufs = (uint8_t **)calloc(numOutputs, sizeof(uint8_t *));

  for (uint32_t i = 0; i < numInputs; i++) {
    ctx->inputs[i] = buildExecTensor(&inputDescs[i], &ctx->inputBufs[i]);
  }
  for (uint32_t i = 0; i < numOutputs; i++) {
    ctx->outputs[i] = buildExecTensor(&outputDescs[i], &ctx->outputBufs[i]);
  }
}

/* Resultado de 1 execucao - qnnRunFrame() nunca imprime o resultado "final"
 * (isso e' decisao de cada modo, ver main()), so' o aviso de mismatch (que e'
 * sempre ruido de diagnostico, independente do modo). */
typedef struct {
  int mismatch; /* 1 = tamanho do frame nao bateu, nada foi executado */
  long execUs;  /* tempo de graphExecute em microsegundos (0 se mismatch) */
} FrameResult;

/* Roda 1 frame: valida tamanho contra o tensor de entrada[0], copia pro
 * buffer, executa o grafo, mede o tempo. NAO decide o que fazer com o
 * resultado (isso fica pro chamador - single-shot trata mismatch como fatal,
 * --loop trata como skip-and-continue, ver main()). */
static FrameResult qnnRunFrame(QnnCtx *ctx, const uint8_t *frameData, size_t frameSize) {
  FrameResult res = {0, 0};
  uint32_t expected = ctx->inputs[0].v1.clientBuf.dataSize;
  if (frameSize != (size_t)expected) {
    fprintf(stderr,
            "[aviso] tamanho do frame (%zu bytes) nao bate com o esperado "
            "pelo tensor de entrada (%u bytes) - confira resolucao/dtype.\n",
            frameSize, expected);
    res.mismatch = 1;
    return res;
  }
  memcpy(ctx->inputBufs[0], frameData, frameSize);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  checkQnn(ctx->qnn->graphExecute(ctx->graphHandle, ctx->inputs, ctx->numInputs,
                                   ctx->outputs, ctx->numOutputs, NULL, NULL),
           "graphExecute");
  clock_gettime(CLOCK_MONOTONIC, &t1);
  res.execUs = (t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000L;
  return res;
}

/* min/max/mean de 1 tensor de saida, sem decodificar deteccoes do YOLO -
 * so' pra confirmar que a saida nao e' lixo/zero/NaN (mesmo espirito de
 * board_test/02_run_inference.py). */
static void computeOutputStats(const Qnn_Tensor_t *outTensor, const uint8_t *buf,
                                size_t *outNumElements, double *outMin, double *outMax,
                                double *outMean) {
  Qnn_DataType_t dtype = TENSOR_GET_DATATYPE(*outTensor);
  size_t elemSize = dtypeElemSize(dtype);
  size_t numElements = outTensor->v1.clientBuf.dataSize / elemSize;

  double minV = 1e300, maxV = -1e300, sum = 0.0;
  for (size_t e = 0; e < numElements; e++) {
    double v;
    switch (dtype) {
      case QNN_DATATYPE_FLOAT_32:
        v = ((const float *)buf)[e];
        break;
      case QNN_DATATYPE_UFIXED_POINT_8:
      case QNN_DATATYPE_UINT_8:
        v = buf[e];
        break;
      case QNN_DATATYPE_SFIXED_POINT_8:
      case QNN_DATATYPE_INT_8:
        v = ((const int8_t *)buf)[e];
        break;
      default:
        v = buf[e]; /* fallback: primeiro byte cru */
        break;
    }
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
    sum += v;
  }
  *outNumElements = numElements;
  *outMin = minV;
  *outMax = maxV;
  *outMean = sum / (double)numElements;
}

/* Imprime stats de TODAS as saidas em `out` - usado como diagnostico
 * completo (stderr no modo --loop, stdout no modo single-shot, ver main()). */
static void printAllOutputStats(const QnnCtx *ctx, FILE *out) {
  for (uint32_t i = 0; i < ctx->numOutputs; i++) {
    size_t numElements;
    double minV, maxV, meanV;
    computeOutputStats(&ctx->outputs[i], ctx->outputBufs[i], &numElements, &minV, &maxV, &meanV);
    fprintf(out, "[infer] saida[%u] '%s': %zu elementos  min=%.4f  max=%.4f  mean=%.4f\n",
            i, TENSOR_GET_NAME(ctx->outputs[i]), numElements, minV, maxV, meanV);
  }
}

/* Libera tudo. systemContextFree() tem que ser a PRIMEIRA coisa aqui (nao no
 * meio do processo, nao antes) - os Qnn_Tensor_t em ctx->inputs/outputs
 * guardam ponteiros (name/dimensions) que apontam pra dentro da memoria do
 * system-context, lidos de novo a cada graphExecute/print de stats durante
 * TODA a vida do processo (inclusive no ultimo frame do modo --loop). Ja foi
 * bug real de use-after-free chamar isso cedo demais. */
static void qnnTeardown(QnnCtx *ctx) {
  ctx->qnnSystem->systemContextFree(ctx->sysCtx);
  ctx->sysCtx = NULL;

  ctx->qnn->contextFree(ctx->context, NULL);
  ctx->qnn->deviceFree(ctx->deviceHandle);
  ctx->qnn->backendFree(ctx->backendHandle);
  ctx->qnn->logFree(ctx->logHandle);

  for (uint32_t i = 0; i < ctx->numInputs; i++) free(ctx->inputBufs[i]);
  for (uint32_t i = 0; i < ctx->numOutputs; i++) free(ctx->outputBufs[i]);
  free(ctx->inputs);
  free(ctx->outputs);
  free(ctx->inputBufs);
  free(ctx->outputBufs);
  free(ctx->binBuffer);

  dlclose(ctx->backendLibHandle);
  dlclose(ctx->systemLibHandle);
}

static void printUsage(const char *argv0) {
  fprintf(stderr, "uso: %s <modelo.bin> <input.raw> [nome_do_grafo]   (modo single-shot)\n",
          argv0);
  fprintf(stderr, "  ou: %s --loop <modelo.bin> [nome_do_grafo]        (modo loop, le' frames do stdin)\n",
          argv0);
}

static int runLoopMode(int argc, char **argv) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }
  const char *binPath = argv[2];
  const char *graphName = (argc >= 4) ? argv[3] : DEFAULT_GRAPH_NAME;

  QnnCtx ctx;
  qnnSetup(&ctx, binPath, graphName);
  fprintf(stderr, "[loop] pronto, aguardando frames no stdin (EOF encerra)...\n");

  uint32_t frameIdx = 0;
  for (;;) {
    uint8_t header[4];
    size_t hgot = readExactStdin(header, 4);
    if (hgot == 0) {
      fprintf(stderr, "[loop] EOF no stdin - encerrando normalmente apos %u frame(s).\n", frameIdx);
      break;
    }
    if (hgot < 4) {
      fprintf(stderr, "[erro] EOF inesperado no meio do header do frame (stream corrompido)\n");
      qnnTeardown(&ctx);
      return 1;
    }

    uint32_t frameSize;
    memcpy(&frameSize, header, 4);
    if (frameSize > MAX_LOOP_FRAME_BYTES) {
      fprintf(stderr, "[erro] frame de %u bytes excede o teto de seguranca (%u) - "
                       "stream corrompido, nao da' mais pra confiar no framing\n",
              frameSize, (unsigned)MAX_LOOP_FRAME_BYTES);
      qnnTeardown(&ctx);
      return 1;
    }

    uint8_t *frameBuf = (uint8_t *)malloc(frameSize);
    size_t pgot = readExactStdin(frameBuf, frameSize);
    if (pgot < frameSize) {
      fprintf(stderr, "[erro] EOF inesperado no meio do payload do frame (stream corrompido)\n");
      free(frameBuf);
      qnnTeardown(&ctx);
      return 1;
    }

    FrameResult res = qnnRunFrame(&ctx, frameBuf, frameSize);
    free(frameBuf);

    if (res.mismatch) {
      printf("ERR tamanho_invalido\n");
      fflush(stdout);
    } else {
      size_t numElements;
      double minV, maxV, meanV;
      computeOutputStats(&ctx.outputs[0], ctx.outputBufs[0], &numElements, &minV, &maxV, &meanV);
      printf("OK %ld %.4f %.4f %.4f\n", res.execUs, minV, maxV, meanV);
      fflush(stdout);
      printAllOutputStats(&ctx, stderr);
    }
    frameIdx++;
  }

  qnnTeardown(&ctx);
  return 0;
}

static int runSingleShotMode(int argc, char **argv) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }
  const char *binPath = argv[1];
  const char *inputRawPath = argv[2];
  const char *graphName = (argc >= 4) ? argv[3] : DEFAULT_GRAPH_NAME;

  QnnCtx ctx;
  qnnSetup(&ctx, binPath, graphName);

  long inputSize = 0;
  void *inputData = readFile(inputRawPath, &inputSize);

  FrameResult res = qnnRunFrame(&ctx, (const uint8_t *)inputData, (size_t)inputSize);
  free(inputData);

  if (res.mismatch) {
    fprintf(stderr, "[erro] tamanho do input.raw nao bate com o esperado pelo tensor "
                     "de entrada - confira resolucao/dtype.\n");
    qnnTeardown(&ctx);
    return 1;
  }

  fprintf(stderr, "[infer] OK - inferencia concluida na HTP em %ld us.\n", res.execUs);
  printAllOutputStats(&ctx, stdout);

  qnnTeardown(&ctx);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--loop") == 0) {
    return runLoopMode(argc, argv);
  }
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }
  return runSingleShotMode(argc, argv);
}
