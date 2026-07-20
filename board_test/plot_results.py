#!/usr/bin/env python3
# =============================================================================
#  plot_results.py
#  Le' os CSVs gerados por monitor.sh (um por modelo, ver experiments/) e monta
#  graficos comparando os modelos: FPS, CPU%, RAM, temperatura CPU/NPU.
#
#  ONDE RODA: no HOST (nao precisa de nada do QAIRT/SDK - so' matplotlib). Puxe
#  os CSVs da placa primeiro (via board_mount ou scp) pra dentro de
#  board_test/experiments/ antes de rodar.
#
#  Ambiente: pyenv virtualenv "placaq6a-plots" (ver .python-version nesta
#  pasta) - nao usa venv/ do board_test (essa e' pra rodar NA placa).
#
#  USO:
#     cd board_test
#     python3 plot_results.py
#  (edite o bloco CONFIG abaixo se mudar nomes/cores de modelo)
# =============================================================================

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# =============================== CONFIG ======================================
# Onde os CSVs do monitor.sh estao (copiados da placa - um arquivo por
# modelo, nome = MODEL_NAME usado em monitor.sh + ".csv").
EXPERIMENTS_DIR = Path("experiments")

# Ordem fixa dos modelos - controla a ordem das cores (nunca cicladas, ver
# skill de dataviz: ordem categorica fixa, nao gerada por rank).
MODELS = ["modelo_260409m-2", "260417_1280_nano", "260420_1280_large"]

MODEL_LABELS = {
    "modelo_260409m-2": "Medio (960)",
    "260417_1280_nano": "Nano (1280)",
    "260420_1280_large": "Large (1280)",
}

# Paleta categorica validada (CVD-safe na ordem abaixo - ver skill de
# dataviz, references/palette.md). 3 modelos = 3 primeiros slots.
COLORS = {
    "modelo_260409m-2": "#2a78d6",  # slot 1 - blue
    "260417_1280_nano": "#008300",  # slot 2 - green
    "260420_1280_large": "#e87ba4",  # slot 3 - magenta
}

OUTDIR = Path("experiments/plots")

# (coluna do csv, rotulo do eixo Y, nome do arquivo de saida)
METRICS = [
    ("fps", "FPS", "fps_comparativo.png"),
    ("cpu_pct", "CPU (%)", "cpu_comparativo.png"),
    ("ram_usado_mb", "RAM usada (MB)", "ram_comparativo.png"),
    ("temp_cpu_c", "Temperatura CPU (°C)", "temp_cpu_comparativo.png"),
    ("temp_npu_c", "Temperatura NPU/DSP (°C)", "temp_npu_comparativo.png"),
]
# =============================================================================


def load_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)


def main():
    data = {}
    for m in MODELS:
        p = EXPERIMENTS_DIR / f"{m}.csv"
        if not p.exists():
            print(f"[aviso] {p} nao encontrado, pulando '{m}'")
            continue
        rows = load_csv(p)
        if not rows:
            print(f"[aviso] {p} vazio, pulando '{m}'")
            continue
        data[m] = rows

    if not data:
        raise SystemExit(
            f"Nenhum CSV encontrado em {EXPERIMENTS_DIR}/ - rode monitor.sh na "
            f"placa e copie os .csv pra ca antes."
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for key, ylabel, fname in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in MODELS:
            if m not in data:
                continue
            rows = data[m]
            x = [int(r["iteracao"]) for r in rows]
            y = [float(r[key]) for r in rows]
            ax.plot(x, y, label=MODEL_LABELS.get(m, m), color=COLORS[m],
                     linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Iteração")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} por iteração — comparação entre modelos")
        ax.legend(frameon=False)
        style_axes(ax)
        fig.tight_layout()
        outpath = OUTDIR / fname
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"[plot] {outpath}")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4.2))
    for ax, (key, ylabel, _) in zip(axes, METRICS):
        labels, values, colors = [], [], []
        for m in MODELS:
            if m not in data:
                continue
            vals = [float(r[key]) for r in data[m]]
            labels.append(MODEL_LABELS.get(m, m))
            values.append(sum(vals) / len(vals))
            colors.append(COLORS[m])
        ax.bar(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=20)
        style_axes(ax)
    fig.suptitle("Médias por modelo")
    fig.tight_layout()
    outpath = OUTDIR / "resumo_medias.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


if __name__ == "__main__":
    main()
