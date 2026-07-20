#!/usr/bin/env bash
# =============================================================================
#  board_mount.sh - monta/desmonta ~/mctech da placa Q6A como pasta LOCAL via
#  sshfs. Da' acesso direto (leitura E escrita) aos artefatos que ja estao la'
#  (board_test/, native_infer/, qairt_runtime/) sem scp manual - e' o MESMO
#  filesystem, so' visto por uma janela de rede: uma mudanca de qualquer lado
#  (voce editando aqui, ou um processo rodando na placa) aparece no outro na
#  hora, sem sincronizacao periodica.
#
#  Requer: sshfs instalado (sudo apt install sshfs) e o Host "q6a" configurado
#  em ~/.ssh/config (chave dedicada id_ed25519_q6a, sem senha - ver
#  ssh-copy-id no README/memoria do projeto). Sem chave, sshfs falha em
#  background/reconexao porque nao tem como digitar senha depois.
#
#  USO (rode no SEU terminal, nao preciso estar aberto por mim - o mount fica
#  ativo ate voce desmontar ou desligar a maquina):
#     ./board_mount.sh mount     # monta a placa em ./board/
#     ./board_mount.sh umount    # desmonta
#     ./board_mount.sh status    # confere se esta montado
# =============================================================================
set -e

# =============================== CONFIG ======================================
# Alias SSH (ver ~/.ssh/config, Host "q6a") - resolve user/IP/chave sozinho.
# ALTERE se usar outro nome de Host ou nao tiver o alias configurado (troque
# por "radxa@192.168.1.119" direto, mas ai' sshfs vai pedir senha).
SSH_HOST="q6a"

# Diretorio remoto (relativo ao $HOME do usuario na placa) que vira ./board/.
BOARD_REMOTE_DIR="mctech"
# =============================================================================

MOUNT_POINT="$(cd "$(dirname "$0")" && pwd)/board"

step_mount() {
  mkdir -p "$MOUNT_POINT"
  if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "[board] ja montado em $MOUNT_POINT"
    return 0
  fi
  sshfs "$SSH_HOST:$BOARD_REMOTE_DIR" "$MOUNT_POINT" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks
  echo "[board] montado em $MOUNT_POINT"
  echo "[board] conteudo:"
  ls "$MOUNT_POINT"
}

step_umount() {
  if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    fusermount3 -u "$MOUNT_POINT" 2>/dev/null || fusermount -u "$MOUNT_POINT"
    echo "[board] desmontado"
  else
    echo "[board] nao estava montado"
  fi
}

step_status() {
  if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "[board] MONTADO em $MOUNT_POINT"
    ls "$MOUNT_POINT"
  else
    echo "[board] nao montado"
  fi
}

case "${1:-status}" in
  mount)  step_mount ;;
  umount) step_umount ;;
  status) step_status ;;
  *)
    echo "uso: $0 [mount|umount|status]"
    exit 1
    ;;
esac
