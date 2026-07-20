#!/usr/bin/env bash
# =============================================================================
#  monitor.sh - roda inferencia repetida sobre um lote FIXO de frames
#  (capturado 1x, reaproveitado em todas as iteracoes - isola o custo da NPU
#  do custo de captura de webcam) e registra, a cada iteracao: FPS, uso de
#  CPU, RAM, temperatura (CPU e NPU/DSP) e estado do CDSP - tudo num CSV, pra
#  comparar modelo por modelo (ver plot_results.py).
#
#  So' pro ambiente board_test/ (via qnn-net-run). NAO existe equivalente
#  valido ainda pro native_infer/: o qnn_infer atual recarrega backend/
#  device/contexto do zero A CADA FRAME (processo novo por chamada), entao
#  benchmarks feitos assim medem overhead de reload repetido, nao o
#  desempenho real da API nativa num uso continuo (camera liga uma vez,
#  modelo carrega uma vez, inferencia roda em loop) - precisaria antes
#  estender native_infer/qnn_infer pra aceitar multiplos frames num so'
#  processo (mantendo o contexto carregado entre eles) pra uma comparacao
#  justa. Ver native_infer/README.md.
#
#  NAO EXISTE um "% de uso da NPU" exposto por esta placa (o unico devfreq
#  com load e' o da GPU; o CDSP so' tem um contador de tempo em baixo-consumo
#  em debugfs, com unidade nao confirmada - nao usamos isso pra nao inventar
#  numero enganoso). Como proxy de atividade da NPU, usamos a temperatura da
#  zona termica "nspss" (NSP SubSystem = o proprio bloco Hexagon/HTP) e o
#  estado do remoteproc "cdsp" (running/crashed - detecta o crash de
#  DMA/VTCM que ja vimos antes, se acontecer de novo).
#
#  ONDE RODA: na PLACA, dentro de board_test/, com o runtime QAIRT ja
#  carregado (source env.sh).
#
#  USO:
#     source ../qairt_runtime/env.sh
#     cd ~/mctech/board_test
#     ./monitor.sh
#  (edite o bloco CONFIG abaixo pro modelo que quer testar - MODEL_NAME vira
#  o nome do arquivo CSV, entao troque a cada modelo diferente)
# =============================================================================
set -u

# =============================== CONFIG ======================================
MODEL_NAME="modelo_260409m-2"                    # so' um rotulo p/ nome do CSV
BIN_PATH="../models/modelo_260409m-2.bin"        # .bin a testar (dir compartilhado)
ITERATIONS=30                                     # quantas rodadas de inferencia
OUTDIR="experiments"                              # onde os CSVs vao parar
# =============================================================================

if [ -z "${QNN_SDK_ROOT:-}" ] || [ -z "${VARIANT:-}" ]; then
  echo "[erro] source ../qairt_runtime/env.sh antes de rodar" >&2
  exit 1
fi
BACKEND="$QNN_SDK_ROOT/lib/$VARIANT/libQnnHtp.so"

mkdir -p "$OUTDIR"
CSV="$OUTDIR/${MODEL_NAME}.csv"

if [ -x venv/bin/python ]; then PY="venv/bin/python"; else PY="python3"; fi

echo "[monitor] capturando lote de frames (1x, reusado em todas as $ITERATIONS iteracoes)..."
"$PY" 01_capture_frames.py
N_FRAMES=$(wc -l < input_list.txt)

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

echo "timestamp,iteracao,fps,ms_por_frame,cpu_pct,ram_usado_mb,temp_cpu_c,temp_npu_c,cdsp_state" > "$CSV"

for i in $(seq 1 "$ITERATIONS"); do
  TMPOUT="/tmp/monitor_out_$$"
  TMPLOG="/tmp/monitor_log_$$.txt"
  stat0=$(read_cpu_stat)
  t0=$(date +%s.%N)
  qnn-net-run --backend "$BACKEND" --retrieve_context "$BIN_PATH" \
    --input_list input_list.txt --output_dir "$TMPOUT" > "$TMPLOG" 2>&1
  ret=$?
  t1=$(date +%s.%N)
  stat1=$(read_cpu_stat)
  rm -rf "$TMPOUT"

  if [ $ret -ne 0 ]; then
    echo "[monitor] iteracao $i FALHOU (qnn-net-run retornou $ret):"
    cat "$TMPLOG" >&2
    rm -f "$TMPLOG"
    continue
  fi
  rm -f "$TMPLOG"

  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b-a}')
  fps=$(awk -v dt="$dt" -v n="$N_FRAMES" 'BEGIN{printf "%.2f", n/dt}')
  ms=$(awk -v dt="$dt" -v n="$N_FRAMES" 'BEGIN{printf "%.2f", (dt/n)*1000}')

  cpu=$(cpu_pct_from_snapshots "$stat0" "$stat1")
  ram=$(read_ram_used_mb)
  tcpu=$(read_temp_max cpu)
  tnpu=$(read_temp_max nspss)
  cdsp=$(read_cdsp_state)
  ts=$(date +%s)

  echo "$ts,$i,$fps,$ms,$cpu,$ram,$tcpu,$tnpu,$cdsp" >> "$CSV"
  echo "[monitor] $i/$ITERATIONS: ${fps} FPS  cpu=${cpu}%  ram=${ram}MB  temp_cpu=${tcpu}C  temp_npu=${tnpu}C  cdsp=${cdsp}"
done

echo "[monitor] concluido -> $CSV"
