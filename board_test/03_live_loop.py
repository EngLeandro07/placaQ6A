#!/usr/bin/env python3
# =============================================================================
#  03_live_loop.py
#  Laco CONTINUO de captura+inferencia (substitui, pra este caso de uso, o
#  fluxo em dois passos de 01_capture_frames.py + 02_run_inference.py - que
#  continuam existindo como smoke test batch, sem mudanca). Por frame:
#     1. captura webcam
#     2. envia JPEG pro streamer/streamer.py (visualizacao ao vivo, ver
#        streamer/README.md) - best-effort, nao trava o loop se o streamer
#        nao estiver de pe
#     3. preprocessa (NCHW/uint8, mesmo formato de 01_capture_frames.py)
#     4. envia pro subprocess `qnn_infer --loop` (native_infer/, ver
#        native_infer/README.md) via stdin/stdout - o modelo e' carregado 1x
#        no inicio deste script, NAO a cada frame
#     5. grava 1 linha no CSV com os tempos de cada etapa + FPS instantaneo
#
#  Nao decodifica deteccoes do YOLO (NMS/grid/ancoras) de proposito - mesmo
#  espirito de 02_run_inference.py e native_infer/qnn_infer: so' confirma que
#  o modelo carrega e roda em uso continuo real, e mede desempenho.
#
#  ONDE RODA: na PLACA, com opencv-python + numpy disponiveis (mesmos
#  requisitos de 01_capture_frames.py) e com native_infer/qnn_infer ja
#  compilado (`cd native_infer && make`, ver README de la').
#
#  USO (na placa, dentro de board_test/):
#     python3 03_live_loop.py
#  Detecta e re-executa sozinho em venv/bin/python se essa venv existir (bloco
#  logo abaixo) - mesmo padrao de run_test.sh/monitor.sh/benchmark_batch.sh,
#  que ja fazem essa deteccao (Python 3.12 da placa e' "externally managed",
#  PEP 668 - ver pre-requisitos em README.md pra criar a venv).
#  (Ctrl+C encerra de forma limpa: fecha webcam, socket do streamer e o
#  subprocess do qnn_infer)
# =============================================================================

import os
import sys
from pathlib import Path

_VENV_PYTHON = Path(__file__).resolve().parent / "venv" / "bin" / "python"
_REEXEC_SENTINEL = "_Q6A_LIVE_LOOP_REEXEC"
# NAO compara sys.executable resolvido contra _VENV_PYTHON resolvido: nesta
# placa venv/bin/python e' so' uma cadeia de symlinks (python -> python3 ->
# /usr/bin/python3) que termina no MESMO binario do Python de sistema - dois
# .resolve() ficam iguais mesmo QUANDO NAO estamos rodando dentro da venv,
# entao a comparacao nunca disparava o re-exec (bug real, encontrado testando
# na placa). A ativacao de venv depende do CAMINHO LITERAL usado pra invocar
# (pra achar o pyvenv.cfg irmao), nao do binario real por tras do symlink -
# por isso usamos um sentinel de variavel de ambiente em vez de comparar
# identidade de interpretador.
if _VENV_PYTHON.exists() and os.environ.get(_REEXEC_SENTINEL) != "1":
    os.environ[_REEXEC_SENTINEL] = "1"
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import socket
import struct
import subprocess
import time

import cv2
import numpy as np


def _shared(key, fallback):
    """Le' 'key' de model.env - fonte unica de verdade compartilhada entre
    container/board_test/native_infer (ver esse arquivo na raiz do repo).
    Ao fazer deploy pra placa via scp, copie model.env pra dentro de
    board_test/ tambem (ver README.md) - senao cai no 'fallback' abaixo."""
    for p in (Path(__file__).resolve().parent / "model.env",
              Path("model.env"),
              Path(__file__).resolve().parent.parent / "model.env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith(key + "="):
                    return line.split("=", 1)[1].strip()
    return fallback


# =============================== CONFIG ======================================
# Mesmo indice de camera de 01_capture_frames.py - ver comentario la' sobre
# por que NAO assumir 0 (encoder de video do SoC pode ocupar /dev/video0/1).
CAM_INDEX = 2

# Resolucao/layout - mesma logica de 01_capture_frames.py (NCHW/RGB, uint8
# bruto, SEM normalizacao - o grafo INT8 ja implantado espera pixel cru).
IMGSZ = int(_shared("IMGSZ", 1280))
LAYOUT = "NCHW"
TO_RGB = True

# Nome do grafo dentro do .bin - ver model.env pra entender de onde vem
# (stem sanitizado de DLC_OUT no passo 02 do pipeline de conversao).
GRAPH_NAME = _shared("GRAPH_NAME", "modelo_fp")

# So' um rotulo (nome do arquivo do CSV) + caminho do .bin em ../models/
# (diretorio COMPARTILHADO com native_infer/, ver CLAUDE.md) - mesmo padrao
# ja usado em 02_run_inference.py/monitor.sh. ALTERE ao trocar de modelo.
MODEL_NAME = "260420_1280_large"
BIN_PATH = f"../models/{MODEL_NAME}.bin"

# Binario compilado de native_infer/ (`cd native_infer && make`).
QNN_INFER_PATH = "../native_infer/qnn_infer"

# Socket do streamer/streamer.py - PRECISA bater com SOCKET_PATH la' (ver
# streamer/README.md pro protocolo). Visualizacao e' best-effort: se o
# streamer nao estiver rodando, o loop de inferencia continua normalmente.
STREAMER_SOCKET_PATH = "/tmp/q6a_streamer.sock"
STREAMER_RECONNECT_INTERVAL_S = 5.0
JPEG_QUALITY = 80

# Onde o CSV frame-a-frame e' gravado. Prefixo "live_" garante zero colisao
# com experiments/<MODEL_NAME>.csv gerado por benchmark_batch.sh (consumido
# por plot_results.py, que so' le' 3 nomes fixos - nunca faz glob).
EXPERIMENTS_DIR = Path("experiments")
CSV_PATH = EXPERIMENTS_DIR / f"live_{MODEL_NAME}.csv"

STATUS_EVERY = 25
# =============================================================================


CSV_HEADER = ("timestamp,frame_idx,fps_inst,t_total_ms,t_capture_ms,"
              "t_preprocess_ms,t_streamer_ms,t_infer_roundtrip_ms,"
              "t_infer_exec_ms,infer_status,out_min,out_max,out_mean\n")


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

    if TO_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = img.astype(np.uint8)

    if LAYOUT == "NCHW":
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
    elif LAYOUT == "NHWC":
        x = np.expand_dims(x, 0)
    else:
        raise ValueError(f"LAYOUT invalido: {LAYOUT}")

    return np.ascontiguousarray(x, dtype=np.uint8)


class StreamerClient:
    """Conexao persistente com streamer/streamer.py (ver protocolo em
    streamer/README.md). Best-effort de proposito: falha de conexao/envio
    NAO derruba o loop de inferencia - so' agenda a proxima tentativa de
    reconexao (gate de tempo, nao tenta a cada frame)."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.sock = None
        self._next_retry = 0.0
        self._try_connect()

    def _try_connect(self):
        now = time.monotonic()
        if now < self._next_retry:
            return
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self.socket_path)
            self.sock = s
            print(f"[live] conectado ao streamer em {self.socket_path}")
        except OSError:
            self.sock = None
            self._next_retry = now + STREAMER_RECONNECT_INTERVAL_S

    def send_jpeg(self, jpeg_bytes: bytes):
        if self.sock is None:
            self._try_connect()
            if self.sock is None:
                return
        try:
            self.sock.sendall(struct.pack("<I", len(jpeg_bytes)) + jpeg_bytes)
        except OSError:
            self._disconnect()

    def _disconnect(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._next_retry = time.monotonic() + STREAMER_RECONNECT_INTERVAL_S

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def start_qnn_infer() -> subprocess.Popen:
    qnn_infer = Path(QNN_INFER_PATH)
    if not qnn_infer.exists():
        raise FileNotFoundError(
            f"{qnn_infer} nao encontrado. compile com 'cd native_infer && make' "
            f"primeiro (na placa, com QAIRT_SDK_ROOT exportado - ver "
            f"native_infer/README.md)."
        )
    bin_path = Path(BIN_PATH)
    if not bin_path.exists():
        raise FileNotFoundError(
            f"modelo nao encontrado: {bin_path}\n"
            f"copie o artefato (.bin) pra dentro de ~/mctech/models/ antes de rodar."
        )
    print(f"[live] iniciando '{qnn_infer} --loop {bin_path} {GRAPH_NAME}'...")
    return subprocess.Popen(
        [str(qnn_infer), "--loop", str(bin_path), GRAPH_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # herdado - logs de setup/diagnostico do C aparecem no mesmo terminal
    )


def run_frame(proc: subprocess.Popen, frame_bytes: bytes):
    """Escreve 1 frame no stdin do qnn_infer --loop (protocolo: 4 bytes LE
    com o tamanho + os bytes) e le' a linha de resposta ("OK ..."/"ERR ...").
    Ver protocolo completo em native_infer/README.md."""
    header = struct.pack("<I", len(frame_bytes))
    proc.stdin.write(header)
    proc.stdin.write(frame_bytes)
    proc.stdin.flush()

    line = proc.stdout.readline()
    if not line:
        raise BrokenPipeError(
            "qnn_infer --loop encerrou inesperadamente (stdout fechou) - "
            "veja os logs de stderr acima pro motivo."
        )
    parts = line.decode().strip().split()
    if parts and parts[0] == "OK":
        _, exec_us, out_min, out_max, out_mean = parts
        return "OK", int(exec_us), float(out_min), float(out_max), float(out_mean)
    return "ERR", 0, 0.0, 0.0, 0.0


def main():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new_csv = not CSV_PATH.exists()
    csv_file = CSV_PATH.open("a")
    if is_new_csv:
        csv_file.write(CSV_HEADER)
        csv_file.flush()

    # proc/streamer sao criados ANTES do try/finally de proposito - mas o
    # try precisa cobrir TUDO a partir daqui (inclusive a abertura da
    # webcam), senao uma falha antes do loop (ex. webcam ocupada) deixa o
    # subprocess do qnn_infer orfao, rodando sozinho com o modelo carregado
    # na NPU - ja foi bug real, encontrado testando na placa.
    proc = start_qnn_infer()
    streamer = StreamerClient(STREAMER_SOCKET_PATH)
    cap = None
    frame_idx = 0
    try:
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(
                f"nao consegui abrir a webcam (indice {CAM_INDEX}). "
                f"confira /dev/video* e se outro processo (ex. 01_capture_frames.py) "
                f"nao esta usando a camera."
            )

        print(f"[live] webcam aberta (indice {CAM_INDEX}), imgsz={IMGSZ}, "
              f"grafo={GRAPH_NAME}. CSV -> {CSV_PATH}. Ctrl+C pra encerrar.")

        while True:
            t0 = time.perf_counter()

            ok, frame_bgr = cap.read()
            if not ok:
                print(f"[live] aviso: falha ao ler frame {frame_idx}, tentando de novo")
                continue
            t_capture = time.perf_counter()

            ok_jpeg, jpeg_buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok_jpeg:
                streamer.send_jpeg(jpeg_buf.tobytes())
            t_streamer = time.perf_counter()

            x = preprocess(frame_bgr)
            t_preprocess = time.perf_counter()

            if proc.poll() is not None:
                raise BrokenPipeError(
                    f"qnn_infer --loop encerrou (returncode={proc.returncode}) - "
                    f"veja os logs acima pro motivo."
                )
            status, exec_us, out_min, out_max, out_mean = run_frame(proc, x.tobytes())
            t_infer = time.perf_counter()

            t_total = t_infer - t0
            fps_inst = 1.0 / t_total if t_total > 0 else 0.0

            csv_file.write(
                f"{time.time():.3f},{frame_idx},{fps_inst:.2f},"
                f"{t_total * 1000:.2f},{(t_capture - t0) * 1000:.2f},"
                f"{(t_preprocess - t_streamer) * 1000:.2f},"
                f"{(t_streamer - t_capture) * 1000:.2f},"
                f"{(t_infer - t_preprocess) * 1000:.2f},{exec_us / 1000:.2f},"
                f"{status},{out_min:.4f},{out_max:.4f},{out_mean:.4f}\n"
            )
            csv_file.flush()

            if frame_idx % STATUS_EVERY == 0:
                print(f"[live] frame {frame_idx}: {fps_inst:.1f} FPS inst. "
                      f"(total={t_total * 1000:.1f}ms infer={exec_us / 1000:.1f}ms) "
                      f"status={status}")

            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[live] Ctrl+C recebido, encerrando...")
    finally:
        if cap is not None:
            cap.release()
        streamer.close()
        csv_file.close()
        if proc.stdin:
            try:
                proc.stdin.close()  # EOF limpo pro qnn_infer - sinaliza desligamento normal
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[live] qnn_infer nao encerrou a tempo, forcando...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"[live] encerrado. CSV -> {CSV_PATH}")


if __name__ == "__main__":
    main()
