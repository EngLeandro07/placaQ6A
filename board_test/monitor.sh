#!/usr/bin/env bash
# =============================================================================
#  monitor.sh - visualizador AO VIVO (tipo `top`) de CPU%, RAM, temperatura
#  (CPU e NPU/DSP) e FPS, pra rodar num terminal separado enquanto
#  03_live_loop.py (captura+inferencia continua) roda em outro.
#
#  Este script NAO roda inferencia nem grava nenhum CSV proprio - so' LE e
#  EXIBE. Antes disso (ate esta mudanca), monitor.sh fazia as duas coisas
#  (rodava qnn-net-run em loop sobre um lote fixo E gravava o CSV de
#  comparacao entre modelos) - essa capacidade foi extraida pra
#  benchmark_batch.sh, sem alteracao de comportamento (ver esse script).
#
#  CPU/RAM/temperatura/estado do CDSP sao amostrados AQUI, localmente, a
#  cada ciclo de refresh (mesmas funcoes de sempre). O FPS vem de fora: quem
#  gera essa metrica frame-a-frame agora e' 03_live_loop.py, que grava em
#  experiments/live_<nome>.csv - este script so' le' a ULTIMA LINHA desse
#  CSV (tail -n1) a cada ciclo, nunca escreve nele.
#
#  NAO EXISTE um "% de uso da NPU" exposto por esta placa (o unico devfreq
#  com load e' o da GPU; o CDSP so' tem um contador de tempo em baixo-consumo
#  em debugfs, com unidade nao confirmada - nao usamos isso pra nao inventar
#  numero enganoso). Como proxy de atividade da NPU, usamos a temperatura da
#  zona termica "nspss" (NSP SubSystem = o proprio bloco Hexagon/HTP) e o
#  estado do remoteproc "cdsp" (running/crashed - detecta o crash de
#  DMA/VTCM que ja vimos antes, se acontecer de novo).
#
#  ONDE RODA: na PLACA, dentro de board_test/. NAO precisa do runtime QAIRT
#  carregado (nao chama qnn-net-run nem nada da NPU - so' le' /proc e /sys).
#
#  USO:
#     cd ~/mctech/board_test
#     ./monitor.sh                        # observa experiments/live.csv
#     ./monitor.sh experiments/live_modelo_260409m-2.csv   # CSV especifico
#  (Ctrl+C encerra e restaura o cursor do terminal)
# =============================================================================
set -u

# =============================== CONFIG ======================================
LIVE_CSV="${1:-experiments/live.csv}"   # CSV que 03_live_loop.py esta escrevendo
REFRESH_INTERVAL=1                       # segundos entre cada atualizacao
STALE_THRESHOLD_S=5                      # acima disso, avisa "parado ha Xs"
# =============================================================================

read_cpu_stat() {
  awk '/^cpu /{print $2,$3,$4,$5,$6,$7,$8}' /proc/stat
}

cpu_pct_from_snapshots() {
  read -r u1 n1 s1 i1 w1 irq1 sirq1 <<< "$1"
  read -r u2 n2 s2 i2 w2 irq2 sirq2 <<< "$2"
  local idle1=$((i1 + w1)) idle2=$((i2 + w2))
  local total1=$((u1 + n1 + s1 + i1 + w1 + irq1 + sirq1))
  local total2=$((u2 + n2 + s2 + i2 + w2 + irq2 + sirq2))
  local dtotal=$((total2 - total1)) didle=$((idle2 - idle1))
  awk -v dt="$dtotal" -v di="$didle" 'BEGIN{ if (dt>0) printf "%.1f", 100*(1-di/dt); else print "0.0" }'
}

read_ram_used_mb() {
  awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%.0f", (t-a)/1024}' /proc/meminfo
}

# max entre as zonas termicas cujo "type" comeca com $1 (ex.: "cpu" -> cpu0..11,
# "nspss" -> nspss0/1 = NPU/DSP). Sem sudo, /sys/class/thermal e' legivel.
read_temp_max() {
  local prefix="$1" max=0
  for z in /sys/class/thermal/thermal_zone*/; do
    local name; name=$(cat "${z}type" 2>/dev/null)
    case "$name" in
      ${prefix}[0-9]*-thermal)
        local t; t=$(($(cat "${z}temp" 2>/dev/null) / 1000))
        [ "$t" -gt "$max" ] && max=$t
        ;;
    esac
  done
  echo "$max"
}

read_cdsp_state() {
  for r in /sys/class/remoteproc/remoteproc*/; do
    if [ "$(cat "${r}name" 2>/dev/null)" = "cdsp" ]; then
      cat "${r}state" 2>/dev/null
      return
    fi
  done
  echo "desconhecido"
}

# Le' a ultima linha de $LIVE_CSV e atualiza FPS_STATUS_LINE (variavel
# global, nao subshell - precisa ser chamada direto, nao dentro de $(...)).
# Colunas do CSV escrito por 03_live_loop.py, nesta ordem (contrato fixo):
#   1=timestamp 2=frame_idx 3=fps_inst 4=t_total_ms ... 13=out_mean
LAST_GOOD_FPS="--"
FPS_STATUS_LINE="-- (aguardando 03_live_loop.py)"

update_fps_status() {
  if [ ! -f "$LIVE_CSV" ]; then
    FPS_STATUS_LINE="-- (aguardando 03_live_loop.py escrever em $LIVE_CSV)"
    return
  fi
  local last_line nf ts
  last_line=$(tail -n 1 "$LIVE_CSV" 2>/dev/null)
  nf=$(awk -F, '{print NF}' <<< "$last_line")
  ts=$(awk -F, '{print $1}' <<< "$last_line")
  # NF==13 sozinho NAO basta pra distinguir uma linha de dados do proprio
  # header (que TAMBEM tem 13 campos, so' que com texto) - valida que o 1o
  # campo (timestamp) e' numerico. Cobre header-only, arquivo vazio, e
  # linha malformada/em escrita parcial - mantem o ultimo valor bom
  # conhecido em vez de mostrar lixo (ja foi bug real: sem essa checagem,
  # um CSV so' com header exibia o literal "fps_inst" como se fosse o FPS).
  if [ "$nf" != "13" ] || ! [[ "$ts" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    FPS_STATUS_LINE="${LAST_GOOD_FPS} (aguardando 1a linha valida em $LIVE_CSV)"
    return
  fi
  local fps now age
  fps=$(awk -F, '{print $3}' <<< "$last_line")
  now=$(date +%s)
  age=$(awk -v now="$now" -v ts="$ts" 'BEGIN{d=now-ts; if(d<0)d=0; printf "%d", d}')
  LAST_GOOD_FPS="$fps"
  if [ "$age" -gt "$STALE_THRESHOLD_S" ]; then
    FPS_STATUS_LINE="${fps} FPS  (parado ha ${age}s - 03_live_loop.py ainda rodando?)"
  else
    FPS_STATUS_LINE="${fps} FPS"
  fi
}

render() {
  tput cup 0 0
  tput ed
  echo "=== board_test/monitor.sh - visualizacao ao vivo (Ctrl+C para sair) ==="
  echo
  printf "  CPU uso:        %s%%\n" "$cpu"
  printf "  RAM usada:      %s MB\n" "$ram"
  printf "  Temp. CPU:      %s C\n" "$tcpu"
  printf "  Temp. NPU/DSP:  %s C  (proxy: zona termica nspss)\n" "$tnpu"
  printf "  CDSP estado:    %s\n" "$cdsp"
  echo
  printf "  FPS (03_live_loop.py):  %s\n" "$FPS_STATUS_LINE"
  echo
  printf "  CSV monitorado: %s\n" "$LIVE_CSV"
  printf "  atualizando a cada %ss\n" "$REFRESH_INTERVAL"
}

cleanup() {
  tput cnorm  # garante que o cursor volta a aparecer, mesmo em Ctrl+C
  echo
  exit 0
}
trap cleanup INT TERM

tput civis  # esconde o cursor (redesenho tipo `top`, sem piscar)
tput clear

stat_prev=$(read_cpu_stat)
while true; do
  sleep "$REFRESH_INTERVAL"

  stat_cur=$(read_cpu_stat)
  cpu=$(cpu_pct_from_snapshots "$stat_prev" "$stat_cur")
  stat_prev="$stat_cur"

  ram=$(read_ram_used_mb)
  tcpu=$(read_temp_max cpu)
  tnpu=$(read_temp_max nspss)
  cdsp=$(read_cdsp_state)
  update_fps_status

  render
done
