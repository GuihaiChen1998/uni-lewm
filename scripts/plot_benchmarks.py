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


if __name__ == "__main__":
    main()
