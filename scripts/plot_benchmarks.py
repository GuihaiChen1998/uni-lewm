"""Export a static publication-style pilot-result figure from audited JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports/benchmarks"
    report = json.loads((directory / "summary.json").read_text())
    if report["missing"]:
        raise RuntimeError("Only plot complete runs")
    rows = report["rows"]
    labels = [
        "Uni-LeWM / CEM",
        "LeWM / CEM",
        "Sub-LeWM / CEM",
        "VLWM / CEM",
        "FlexiWorld / CEM",
        "FlexiWorld / Direct",
        "Dreamer V1 / offline",
        "Dreamer V2 / offline",
        "Dreamer V3 / offline",
        "Random",
        "Oracle (privileged)",
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(rows))
    colors = ["#2563eb"] + ["#94a3b8"] * (len(rows) - 2) + ["#cbd5e1"]
    means = [r["mean_success_rate"] * 100 for r in rows]
    ax.barh(y, means, color=colors, height=0.6)
    for index, row in enumerate(rows):
        counts = row["success_counts"]
        offsets = np.linspace(-0.14, 0.14, len(counts)) if len(counts) > 1 else [0]
        ax.scatter(
            np.array(counts) * 5,
            index + np.array(offsets),
            s=23,
            facecolors="white",
            edgecolors="#0f172a",
            zorder=4,
        )
        ax.text(103, index, "/".join(str(c) for c in counts), va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Success rate (%)   •   dots: individual seeds")
    ax.xaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    ax.set_title(
        "Toy-scale pilot: 20 fixed start/goal tasks", loc="left", fontweight="bold", pad=24
    )
    ax.text(
        0,
        1.015,
        "1 environment · 384 training transitions · 1,000 updates · 3 seeds",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    ax.text(103, -0.8, "Successes / 20", fontsize=8)
    fig.text(
        0.02,
        0.015,
        "Adapted implementations, not original-paper benchmarks. Dreamer uses offline hindsight goal/reward supervision.",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(directory / "success_rates.png", dpi=180)
    fig.savefig(directory / "success_rates.svg")
    plt.close(fig)

    matrix = []
    for row in rows:
        count = row["expected_seeds"]
        trials = [
            json.loads((directory / f"{row['method']}-seed{s}.json").read_text())
            for s in range(count)
        ]
        matrix.append(
            [sum(trial["results"][t]["reached"] for trial in trials) / count for t in range(20)]
        )
    fig, ax = plt.subplots(figsize=(11, 6))
    heat = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_xticks(np.arange(20), np.arange(20))
    ax.set_xlabel("Fixed task ID (same start/goal pairs for every method)")
    ax.set_title("Per-task success across seeds", loc="left", fontweight="bold", pad=18)
    colorbar = fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_ticks([0, 1 / 3, 2 / 3, 1], labels=["0/3", "1/3", "2/3", "3/3"])
    colorbar.set_label("Successful runs (oracle: one deterministic run)")
    ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.02,
        0.015,
        "Toy-scale/offline adaptations. Three seeds on the same 20 tasks are not 60 independent tasks.",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(directory / "task_matrix.png", dpi=180)
    fig.savefig(directory / "task_matrix.svg")


if __name__ == "__main__":
    main()
