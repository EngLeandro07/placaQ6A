/* =============================================================================
 *  qnn_infer.c
 *  Cliente NATIVO em C puro para a API QNN (Qualcomm) - roda o context-binary
 *  (.bin) gerado pelo passo 04 direto na HTP, sem passar pelo qnn-net-run
 *  (CLI) nem por wrapper Python/subprocess. So' a API QNN, chamada direto.
 *
 *  ONDE RODA: na PLACA, com o runtime QAIRT carregado (source env.sh), pra
 *  que libQnnHtp.so/libQnnSystem.so estejam no LD_LIBRARY_PATH.
 *
 *  SEQUENCIA (extraida do QnnSampleApp.cpp oficial do SDK, mesma ordem):
 *    dlopen(libQnnSystem.so) -> QnnSystemInterface_getProviders
 *    dlopen(libQnnHtp.so)    -> QnnInterface_getProviders
 *    logCreate -> backendCreate -> deviceCreate
 *    systemContextCreate -> systemContextGetBinaryInfo (le' metadados do .bin:
 *      nomes/shapes/dtypes dos grafos e tensores, sem precisar saber de
 *      antemao o formato do modelo) -> systemContextFree
 *    contextCreateFromBinary (carrega o .bin na HTP)
 *    graphRetrieve (pega o grafo pelo nome)
 *    graphExecute  (roda 1 inferencia sobre o .raw de entrada)
 *    contextFree -> deviceFree -> backendFree -> logFree
 *
 *  NAO conserta o crash de DMA/VTCM ja diagnosticado no CDSP da placa - so'
 *  troca COMO a inferencia e' chamada (nativo em vez de CLI). Se o problema
 *  de firmware/FastRPC nao estiver resolvido, isto vai falhar do mesmo jeito
 *  (mas com mensagens de erro potencialmente mais detalhadas via QnnError).
 *
 *  USO:
 *     ./qnn_infer <modelo.bin> <input.raw> [nome_do_grafo]
 *
 *  Compilar na propria placa (ver Makefile / README.md).
 * =============================================================================
 */

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

/* =============================== CONFIG ==================================== */
/* Nome das libs do backend HTP e do system-lib. Resolvidos via LD_LIBRARY_PATH
 * (setado pelo env.sh do runtime QAIRT) - nao precisam de caminho absoluto. */
#define BACKEND_LIB_NAME "libQnnHtp.so"
#define SYSTEM_LIB_NAME  "libQnnSystem.so"

/* Nome do grafo default, se nao passado por argv (3o parametro). O Makefile
 * le' GRAPH_NAME de model.env (raiz do repo, fonte unica de verdade) e passa
 * via -D na compilacao; se nao passado (compilado fora do Makefile), cai
 * neste fallback. Ver model.env pra entender de onde esse nome vem. */
#ifndef DEFAULT_GRAPH_NAME
#define DEFAULT_GRAPH_NAME "modelo_fp"
#endif
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

/* ---- Helpers de dispatch de versao do Qnn_Tensor_t (V1 ou V2) --------------
 * O .bin pode descrever tensores em V1 ou V2 (QnnSystemContext_GraphInfoV*).
 * So' precisamos de um subconjunto pequeno de campos (id/name/dataType/
 * rank/dimensions/memType/clientBuf) - os dois tem os mesmos primeiros
 * campos com o mesmo layout, entao os macros abaixo dao conta dos dois. */
#define TENSOR_GET_NAME(t)      (((t).version == QNN_TENSOR_VERSION_1) ? (t).v1.name : (t).v2.name)
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

  t.v1.id = 0;
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

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "uso: %s <modelo.bin> <input.raw> [nome_do_grafo]\n", argv[0]);
    return 1;
  }
  const char *binPath = argv[1];
  const char *inputRawPath = argv[2];
  const char *graphName = (argc >= 4) ? argv[3] : DEFAULT_GRAPH_NAME;

  /* ---- 1. Carrega libQnnSystem.so e libQnnHtp.so, resolve os providers ---- */
  void *systemLibHandle = mustDlopen(SYSTEM_LIB_NAME);
  void *backendLibHandle = mustDlopen(BACKEND_LIB_NAME);

  typedef Qnn_ErrorHandle_t (*GetProvidersFn_t)(const QnnInterface_t ***, uint32_t *);
  typedef Qnn_ErrorHandle_t (*GetSystemProvidersFn_t)(const QnnSystemInterface_t ***, uint32_t *);

  GetProvidersFn_t getProviders =
      (GetProvidersFn_t)mustDlsym(backendLibHandle, "QnnInterface_getProviders");
  GetSystemProvidersFn_t getSystemProviders =
      (GetSystemProvidersFn_t)mustDlsym(systemLibHandle, "QnnSystemInterface_getProviders");

  const QnnInterface_t **providers = NULL;
  uint32_t numProviders = 0;
  checkQnn(getProviders(&providers, &numProviders), "QnnInterface_getProviders");

  const QNN_INTERFACE_VER_TYPE *qnn = NULL;
  for (uint32_t i = 0; i < numProviders; i++) {
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
      qnn = &providers[i]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!qnn) {
    fprintf(stderr, "[erro] nenhum provider QNN compativel (API major %d)\n",
            QNN_API_VERSION_MAJOR);
    return 1;
  }

  const QnnSystemInterface_t **sysProviders = NULL;
  uint32_t numSysProviders = 0;
  checkQnn(getSystemProviders(&sysProviders, &numSysProviders),
           "QnnSystemInterface_getProviders");
  const QNN_SYSTEM_INTERFACE_VER_TYPE *qnnSystem = NULL;
  for (uint32_t i = 0; i < numSysProviders; i++) {
    if (sysProviders[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR) {
      qnnSystem = &sysProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!qnnSystem) {
    fprintf(stderr, "[erro] nenhum system-provider compativel\n");
    return 1;
  }

  /* ---- 2. log / backend / device -------------------------------------- */
  Qnn_LogHandle_t logHandle = NULL;
  checkQnn(qnn->logCreate(logCallback, QNN_LOG_LEVEL_WARN, &logHandle), "logCreate");

  Qnn_BackendHandle_t backendHandle = NULL;
  checkQnn(qnn->backendCreate(logHandle, NULL, &backendHandle), "backendCreate");

  Qnn_DeviceHandle_t deviceHandle = NULL;
  checkQnn(qnn->deviceCreate(logHandle, NULL, &deviceHandle), "deviceCreate");

  printf("[infer] backend/device criados. carregando %s...\n", binPath);

  /* ---- 3. le' o .bin e extrai metadados via system-context ------------- */
  long binSize = 0;
  void *binBuffer = readFile(binPath, &binSize);

  QnnSystemContext_Handle_t sysCtx = NULL;
  checkQnn(qnnSystem->systemContextCreate(&sysCtx), "systemContextCreate");

  const QnnSystemContext_BinaryInfo_t *binaryInfo = NULL;
  Qnn_ContextBinarySize_t binaryInfoSize = 0;
  checkQnn(qnnSystem->systemContextGetBinaryInfo(sysCtx, binBuffer, (uint64_t)binSize,
                                                  &binaryInfo, &binaryInfoSize),
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
  printf("[infer] %s tem %u grafo(s)\n", binPath, numGraphs);

  /* acha o grafo pelo nome e copia a descricao dos tensores de I/O (a
   * memoria do system-context sera liberada depois, entao copiamos os
   * Qnn_Tensor_t - dimensions/name apontam pra dentro do binBuffer, que
   * continua vivo, entao esses ponteiros continuam validos). */
  QnnSystemContext_GraphInfo_t *targetGraph = NULL;
  for (uint32_t i = 0; i < numGraphs; i++) {
    const char *name = (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
                            ? graphs[i].graphInfoV1.graphName
                        : (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2)
                            ? graphs[i].graphInfoV2.graphName
                            : graphs[i].graphInfoV3.graphName;
    printf("[infer]   grafo[%u]: %s\n", i, name);
    if (strcmp(name, graphName) == 0) targetGraph = &graphs[i];
  }
  if (!targetGraph) {
    fprintf(stderr, "[erro] grafo '%s' nao encontrado no .bin\n", graphName);
    return 1;
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
  printf("[infer] grafo '%s': %u entrada(s), %u saida(s)\n", graphName, numInputs, numOutputs);

  /* ---- 4. cria o contexto a partir do binario (carrega na HTP) --------- */
  Qnn_ContextHandle_t context = NULL;
  checkQnn(qnn->contextCreateFromBinary(backendHandle, deviceHandle, NULL, binBuffer,
                                        (Qnn_ContextBinarySize_t)binSize, &context, NULL),
           "contextCreateFromBinary");

  Qnn_GraphHandle_t graphHandle = NULL;
  checkQnn(qnn->graphRetrieve(context, graphName, &graphHandle), "graphRetrieve");

  qnnSystem->systemContextFree(sysCtx);
  sysCtx = NULL;

  printf("[infer] contexto carregado na HTP, grafo '%s' recuperado.\n", graphName);

  /* ---- 5. monta tensores de execucao e carrega o input.raw -------------- */
  Qnn_Tensor_t *inputs = (Qnn_Tensor_t *)calloc(numInputs, sizeof(Qnn_Tensor_t));
  Qnn_Tensor_t *outputs = (Qnn_Tensor_t *)calloc(numOutputs, sizeof(Qnn_Tensor_t));
  uint8_t **inputBufs = (uint8_t **)calloc(numInputs, sizeof(uint8_t *));
  uint8_t **outputBufs = (uint8_t **)calloc(numOutputs, sizeof(uint8_t *));

  for (uint32_t i = 0; i < numInputs; i++) {
    inputs[i] = buildExecTensor(&inputDescs[i], &inputBufs[i]);
  }
  for (uint32_t i = 0; i < numOutputs; i++) {
    outputs[i] = buildExecTensor(&outputDescs[i], &outputBufs[i]);
  }

  /* input .raw -> buffer do primeiro tensor de entrada (modelo de 1 entrada,
   * como o YOLO exportado no passo 01/02 deste projeto). */
  long inputSize = 0;
  void *inputData = readFile(inputRawPath, &inputSize);
  uint32_t expectedSize = inputs[0].v1.clientBuf.dataSize;
  if ((long)expectedSize != inputSize) {
    fprintf(stderr,
            "[erro] tamanho do input.raw (%ld bytes) nao bate com o esperado "
            "pelo tensor de entrada (%u bytes) - confira resolucao/dtype.\n",
            inputSize, expectedSize);
    return 1;
  }
  memcpy(inputBufs[0], inputData, (size_t)inputSize);
  free(inputData);

  /* ---- 6. executa -------------------------------------------------------- */
  printf("[infer] executando graphExecute...\n");
  checkQnn(qnn->graphExecute(graphHandle, inputs, numInputs, outputs, numOutputs, NULL, NULL),
           "graphExecute");
  printf("[infer] OK - inferencia concluida na HTP.\n");

  /* ---- 7. estatisticas basicas de saida (min/max/mean), sem decodificar
   * as deteccoes do YOLO - mesmo espirito do board_test/02_run_inference.py */
  for (uint32_t i = 0; i < numOutputs; i++) {
    Qnn_DataType_t dtype = TENSOR_GET_DATATYPE(outputs[i]);
    size_t elemSize = dtypeElemSize(dtype);
    size_t numElements = outputs[i].v1.clientBuf.dataSize / elemSize;
    uint8_t *buf = outputBufs[i];

    double minV = 1e300, maxV = -1e300, sum = 0.0;
    for (size_t e = 0; e < numElements; e++) {
      double v;
      switch (dtype) {
        case QNN_DATATYPE_FLOAT_32:
          v = ((float *)buf)[e];
          break;
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UINT_8:
          v = buf[e];
          break;
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_INT_8:
          v = ((int8_t *)buf)[e];
          break;
        default:
          v = buf[e]; /* fallback: primeiro byte cru */
          break;
      }
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
      sum += v;
    }
    printf("[infer] saida[%u] '%s': %zu elementos  min=%.4f  max=%.4f  mean=%.4f\n", i,
           TENSOR_GET_NAME(outputs[i]), numElements, minV, maxV, sum / (double)numElements);
  }

  /* ---- 8. cleanup ---------------------------------------------------------*/
  qnn->contextFree(context, NULL);
  qnn->deviceFree(deviceHandle);
  qnn->backendFree(backendHandle);
  qnn->logFree(logHandle);

  for (uint32_t i = 0; i < numInputs; i++) free(inputBufs[i]);
  for (uint32_t i = 0; i < numOutputs; i++) free(outputBufs[i]);
  free(inputs);
  free(outputs);
  free(inputBufs);
  free(outputBufs);
  free(binBuffer);

  dlclose(backendLibHandle);
  dlclose(systemLibHandle);

  return 0;
}
