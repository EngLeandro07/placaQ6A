#!/usr/bin/env python3
# =============================================================================
#  streamer.py
#  Processo dedicado que recebe frames JPEG de UM produtor por vez (via Unix
#  socket local) e os serve como MJPEG-over-HTTP para qualquer navegador na
#  mesma LAN da placa. E' o "nivel abaixo" compartilhado entre board_test/
#  (Python) e um futuro produtor em C: o protocolo do socket e' agnostico de
#  linguagem (framing binario simples, ver README.md deste diretorio) - so'
#  quem hoje tem acesso a camera (board_test/03_live_loop.py, via cv2) fala
#  com ele, mas nada aqui e' especifico de Python.
#
#  ONDE RODA: na PLACA (nao precisa de QAIRT nem de opencv - so' stdlib).
#  A placa nao tem HDMI funcional, entao a visualizacao e' sempre pelo
#  navegador do HOST, apontando pro IP da placa (ex. http://192.168.1.119:8080/).
#
#  USO (na placa):
#     python3 streamer.py
#  (roda em primeiro plano; Ctrl+C encerra e remove o socket)
# =============================================================================

import os
import socket
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# =============================== CONFIG ======================================
# Socket Unix local onde produtores (board_test/03_live_loop.py hoje; qualquer
# outro processo C/Python amanha) conectam e escrevem frames JPEG. Fica fora
# do repo de proposito (e' um artefato de runtime, nao um arquivo versionado).
SOCKET_PATH = "/tmp/q6a_streamer.sock"

# Porta HTTP onde o navegador do HOST acessa "/" (pagina) e "/stream" (MJPEG).
# SEM autenticacao/TLS - aceitavel em LAN de dev, NAO exponha alem disso.
HTTP_PORT = 8080

# Teto de sanidade pro tamanho de 1 frame JPEG. Um N maior que isso no header
# indica dessincronia de protocolo (nao um frame legitimo grande) - encerra
# so' a conexao daquele produtor, sem derrubar o processo inteiro.
MAX_FRAME_BYTES = 5 * 1024 * 1024

BOUNDARY = "frame"
# =============================================================================


class FrameBroadcaster:
    """Guarda o ultimo frame recebido e acorda todos os viewers HTTP quando
    um frame novo chega. Sem fila por viewer de proposito: cada viewer sempre
    ve o frame mais recente, descartando frames intermediarios se ele for
    mais lento que o produtor - e' o comportamento correto pra MJPEG ao vivo
    (nao queremos acumular atraso)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._version = 0

    def update(self, frame: bytes):
        with self._cond:
            self._frame = frame
            self._version += 1
            self._cond.notify_all()

    def get_next(self, last_version: int):
        """Bloqueia ate existir um frame com versao > last_version."""
        with self._cond:
            while self._frame is None or self._version == last_version:
                self._cond.wait()
            return self._version, self._frame


def _recv_exact(conn: socket.socket, n: int):
    """Le exatamente n bytes de conn, tratando recv() parcial (AF_UNIX
    SOCK_STREAM e' orientado a fluxo, igual TCP - um recv() pode retornar
    menos do que foi pedido). Retorna None em EOF limpo (produtor fechou)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _make_listening_socket() -> socket.socket:
    """Faz bind no SOCKET_PATH, cuidando do caso de um socket 'grudado' de
    uma execucao anterior que morreu sem limpar o arquivo. Tenta conectar
    primeiro: se conectar, ja existe uma instancia viva (aborta, nao mata
    silenciosamente um processo legitimo); se falhar, o arquivo e' lixo de
    uma execucao anterior e e' seguro remover."""
    if os.path.exists(SOCKET_PATH):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(SOCKET_PATH)
            probe.close()
            print(f"[erro] ja existe uma instancia do streamer rodando em "
                  f"{SOCKET_PATH} (conectei com sucesso). Encerre-a antes de "
                  f"subir outra.", file=sys.stderr)
            sys.exit(1)
        except OSError:
            probe.close()
            os.unlink(SOCKET_PATH)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH)
    sock.listen(1)
    return sock


def ingest_loop(broadcaster: FrameBroadcaster, listen_sock: socket.socket):
    """Thread dedicada: aceita 1 produtor por vez (so' existe 1 camera fisica
    na placa) e le frames em loop. Se o produtor cair/reconectar, os viewers
    continuam vendo o ultimo frame congelado ate o proximo update().

    Recebe o listen_sock ja' pronto (nao cria ele mesma) de proposito: a
    deteccao de "ja existe uma instancia rodando" em _make_listening_socket()
    chama sys.exit(1) em caso de conflito, e isso so' encerra o PROCESSO
    inteiro se rodar na thread principal - dentro de uma thread secundaria
    (como esta), sys.exit() so' mataria essa thread, deixando o processo
    seguir e tentar (e falhar, com traceback feio) dar bind na porta HTTP
    tambem. Por isso _make_listening_socket() e' chamado em main(), ANTES de
    subir esta thread."""
    while True:
        conn, _ = listen_sock.accept()
        print("[streamer] produtor conectado")
        try:
            while True:
                header = _recv_exact(conn, 4)
                if header is None:
                    break
                (n,) = struct.unpack("<I", header)
                if n > MAX_FRAME_BYTES:
                    print(f"[streamer] aviso: frame de {n} bytes excede "
                          f"MAX_FRAME_BYTES ({MAX_FRAME_BYTES}) - desync de "
                          f"protocolo, encerrando esta conexao")
                    break
                payload = _recv_exact(conn, n)
                if payload is None:
                    break
                broadcaster.update(payload)
        except OSError as e:
            print(f"[streamer] aviso: erro lendo do produtor: {e}")
        finally:
            conn.close()
            print("[streamer] produtor desconectado - aguardando novo "
                  "produtor (ultimo frame continua sendo servido aos viewers)")


INDEX_HTML = b"""<!doctype html>
<html>
<head><title>Q6A - stream ao vivo</title></head>
<body style="margin:0;background:#111">
<img src="/stream" style="width:100%;height:auto;display:block">
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    broadcaster: FrameBroadcaster = None  # setado em main() antes do serve_forever

    def log_message(self, fmt, *args):
        pass  # silencia o log de acesso padrao (ruidoso pra stream continuo)

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
        elif self.path == "/stream":
            self._stream_mjpeg()
        else:
            self.send_response(404)
            self.end_headers()

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                          f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        last_version = 0
        try:
            while True:
                last_version, frame = self.broadcaster.get_next(last_version)
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer fechou a aba/conexao - encerra a thread desse cliente


def main():
    # Feito na thread principal DE PROPOSITO (nao dentro de ingest_loop): se
    # ja' houver uma instancia rodando, isso chama sys.exit(1) - precisa
    # abortar o processo inteiro antes de tentar qualquer bind, nao so' uma
    # thread secundaria (ver comentario em ingest_loop).
    listen_sock = _make_listening_socket()
    print(f"[streamer] ouvindo produtores em {SOCKET_PATH}")

    broadcaster = FrameBroadcaster()
    StreamHandler.broadcaster = broadcaster

    threading.Thread(target=ingest_loop, args=(broadcaster, listen_sock), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), StreamHandler)
    print(f"[streamer] HTTP em http://0.0.0.0:{HTTP_PORT}/ (Ctrl+C para sair)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[streamer] encerrando...")
    finally:
        server.server_close()
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
