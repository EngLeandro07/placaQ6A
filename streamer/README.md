# Streamer — visualização MJPEG ao vivo, compartilhada entre C e Python

Processo dedicado, separado dos ambientes de inferência (`board_test/`,
`native_infer/`), que recebe frames JPEG de **um produtor por vez** via um
Unix socket local e os serve como **MJPEG-over-HTTP** para qualquer
navegador na mesma LAN da placa. Existe porque a Q6A não tem HDMI
funcional — a única forma de "ver" o que a câmera está capturando é pelo
navegador de outra máquina.

É o "nível abaixo" pedido para não duplicar lógica de streaming em C e em
Python: o **protocolo do socket é agnóstico de linguagem** — hoje só
`board_test/03_live_loop.py` (Python, via `cv2`) fala com ele, porque é o
único lugar do repo com acesso à câmera, mas qualquer processo em C que
algum dia capture frames (o `native_infer/qnn_infer` atual não captura —
só recebe tensores já pré-processados) pode falar o mesmo protocolo sem
precisar reimplementar HTTP/multipart.

## Protocolo do socket (produtor → streamer)

- Caminho: `/tmp/q6a_streamer.sock` (`SOCKET_PATH` em `streamer.py`) — fica
  fora do repo, é um artefato de runtime, não um arquivo versionado.
- Um produtor conecta (`AF_UNIX`, `SOCK_STREAM`) e escreve, para cada
  frame, em sequência:
  1. **4 bytes little-endian** (`uint32`) = `N`, o tamanho em bytes do JPEG.
  2. **N bytes** do JPEG (ex. saída de `cv2.imencode('.jpg', frame)` em C).
- Sem handshake, sem resposta do streamer — é só um relay de bytes opacos
  (ele nunca decodifica o JPEG).
- Teto de sanidade: `N` acima de `MAX_FRAME_BYTES` (5 MB) é tratado como
  dessincronia de protocolo — o streamer encerra **só aquela conexão**,
  sem derrubar o processo.
- **Só um produtor por vez** (só existe uma câmera física na placa). Se o
  produtor cair ou fechar a conexão, o streamer volta a aceitar uma nova
  conexão automaticamente — os viewers continuam vendo o **último frame
  recebido, congelado**, até um novo produtor conectar e mandar frame novo.
- Exemplo mínimo em Python (é basicamente o que `03_live_loop.py` faz):
  ```python
  import socket, struct
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.connect("/tmp/q6a_streamer.sock")
  jpeg_bytes = ...  # cv2.imencode(".jpg", frame)[1].tobytes()
  sock.sendall(struct.pack("<I", len(jpeg_bytes)) + jpeg_bytes)
  ```
  Em C o equivalente é `connect()` num `AF_UNIX`/`SOCK_STREAM`, escrever 4
  bytes little-endian com o tamanho e depois os bytes do JPEG — sem
  nenhuma dependência além de sockets POSIX padrão.

## Lado HTTP (streamer → navegador)

- `GET /` — página HTML mínima com `<img src="/stream">`.
- `GET /stream` — `multipart/x-mixed-replace; boundary=frame`, cada parte:
  ```
  --frame\r\n
  Content-Type: image/jpeg\r\n
  Content-Length: <N>\r\n
  \r\n
  <N bytes de JPEG>\r\n
  ```
- Múltiplos viewers simultâneos são suportados — cada um sempre recebe o
  frame mais recente (nunca acumula fila/atraso; se o viewer for mais
  lento que a captura, ele só "perde" os frames intermediários).
- Antes do primeiro frame chegar de algum produtor, a conexão do viewer
  fica bloqueada esperando (o navegador mostra "carregando"), sem nenhum
  placeholder gerado.
- **Sem autenticação nem TLS** — aceitável para uso em LAN de
  desenvolvimento; não exponha essa porta além disso.

## Rodar

Na placa:
```bash
python3 streamer.py
```
Roda em primeiro plano (Ctrl+C encerra e remove o socket). Só usa a
biblioteca padrão do Python (`socket`, `struct`, `threading`,
`http.server`) — não precisa de venv nem de nenhum pacote extra.

No navegador do host, com a placa em `192.168.1.119` (ajuste o IP):
```
http://192.168.1.119:8080/
```

## Deploy

Mesmo padrão de `board_test/`/`native_infer/` (ver `CLAUDE.md`, seção
"Environments & shared data" e "`board_mount.sh`"):
```bash
./board_mount.sh mount        # na raiz do repo, uma vez por sessão
cp -r streamer board/
```
ou via `scp -r streamer radxa@<ip-da-placa>:~/mctech/streamer`.

## Socket "grudado" de uma execução anterior

Se `streamer.py` crashar sem limpar `/tmp/q6a_streamer.sock`, a próxima
execução detecta isso automaticamente: tenta `connect()` no socket
existente — se conseguir, já há uma instância viva rodando (aborta com
erro claro em vez de subir uma segunda instância); se falhar, o arquivo é
lixo de uma execução anterior e é removido antes de criar o novo socket.
