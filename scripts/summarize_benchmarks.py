"""Validate raw task results, archive configs/logs, and generate the result table."""

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

from dreamer_data import DATA, ROOT, TASKS, digest, grid_class

ROWS = [
    ("uni_lewm", "Uni-LeWM + CEM"),
    ("lewm", "LeWM + CEM (official modules, toy scale)"),
    ("sub_lewm", "Sub-LeWM + CEM (Sub-JEPA regularizer)"),
    ("vlwm", "VLWM + CEM (independent paper adaptation)"),
    ("flexiworld", "FlexiWorld + CEM (discrete adaptation)"),
    ("flexiworld-direct", "FlexiWorld Direct (same checkpoints)"),
    ("dreamerv1", "Dreamer V1 (offline goal-conditioned)"),
    ("dreamerv2", "Dreamer V2 (offline goal-conditioned)"),
    ("dreamerv3", "Dreamer V3 (offline goal-conditioned)"),
    ("random", "Random"),
    ("oracle_shortest_path", "Oracle shortest path (privileged)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "work/benchmark20")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports/benchmarks")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    tasks = json.loads(TASKS.read_text())["tasks"]
    replay = grid_class()(9, 32, 3001)
    summary, missing = [], []
    for method, label in ROWS:
        records, training = [], []
        seeds = [0] if method == "oracle_shortest_path" else [0, 1, 2]
        for seed in seeds:
            file = args.reports / f"{method}-seed{seed}.json"
            if not file.exists():
                missing.append(file.name)
                continue
            report = json.loads(file.read_text())
            assert report["tasks_sha256"] == digest(TASKS), file
            assert report["dataset_sha256"] == digest(DATA), file
            assert len(report["results"]) == report["episodes"] == 20, file
            assert sum(r["reached"] for r in report["results"]) == report["success_count"], file
            for expected, actual in zip(tasks, report["results"]):
                assert (
                    actual["start"] == expected["start"] and actual["goal"] == expected["goal"]
                ), file
                assert actual["steps"] <= 30, file
                assert len(actual["actions"]) == actual["steps"], file
                replay.reset(tuple(expected["start"]))
                first_hit = None
                for step, action in enumerate(actual["actions"], 1):
                    assert 0 <= action < 4, file
                    replay.step(action)
                    if replay.position == tuple(expected["goal"]) and first_hit is None:
                        first_hit = step
                assert actual["reached"] == (first_hit is not None), (file, actual["episode"])
                if "first_hit_step" in actual:
                    assert first_hit == actual["first_hit_step"], (file, actual["episode"])
            records.append(report)
            model = method.removesuffix("-direct")
            run = args.run_root / f"{model}-seed{seed}"
            if (run / "training.json").exists():
                meta = json.loads((run / "training.json").read_text())
                config = meta.get("config", meta)
                assert config["steps"] == 1000, run
                assert config["dataset_sha256"] == digest(DATA), run
                checkpoint = run / ("last.pkl" if model.startswith("dreamerv") else "last.pt")
                assert report["checkpoint_sha256"] == digest(checkpoint), file
                assert report["training_seed"] == seed, file
                if model in {"dreamerv1", "dreamerv2"}:
                    assert "initialization_seed_control" in meta, run
                training.append(meta)
                dest = args.reports / "training" / run.name
                dest.mkdir(parents=True, exist_ok=True)
                for name in ("config.json", "training.json", "metrics.json"):
                    shutil.copy2(run / name, dest / name)
        rates = [r["success_rate"] for r in records]
        summary.append(
            {
                "method": method,
                "label": label,
                "completed_seeds": len(rates),
                "expected_seeds": len(seeds),
                "success_counts": [r["success_count"] for r in records],
                "mean_success_rate": statistics.mean(rates) if rates else None,
                "sample_std": statistics.stdev(rates) if len(rates) > 1 else None,
                "parameters": training[0]["parameters"]
                if training
                else 0
                if rates and method in {"random", "oracle_shortest_path"}
                else None,
                "per_seed_false_stops": [r.get("false_stop_count", 0) for r in records],
            }
        )
    if missing and not args.allow_partial:
        raise RuntimeError("Incomplete experiments: " + ", ".join(missing))
    payload = {
        "protocol": "docs/BENCHMARKS.md",
        "updates_per_model": 1000,
        "training_seeds": [0, 1, 2],
        "task_count": 20,
        "distinct_environments": 1,
        "dataset_sha256": digest(DATA),
        "tasks_sha256": digest(TASKS),
        "missing": missing,
        "rows": summary,
    }
    (args.reports / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.reports / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# 20-task toy comparison",
        "",
        "**Toy-scale/offline adaptations, not original-paper benchmark reproductions.**",
        "One grid environment; 20 fixed start/goal pairs; 30-action limit; 1,000 updates; three training seeds.",
        "See [full protocol](../../docs/BENCHMARKS.md) for sources and differences.",
        "Muse's separate Kaggle run and ablation are discussed in the [issue #1 review](../muse_issue1_review.md).",
        "",
        "| Method | Successes / 20 by seed | Mean ± sample SD | Parameters |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary:
        counts = ", ".join(map(str, row["success_counts"])) or "pending"
        if row["completed_seeds"] < row["expected_seeds"]:
            counts += " (incomplete)"
        rate = (
            "pending"
            if row["mean_success_rate"] is None
            else f"{row['mean_success_rate'] * 100:.1f}%"
        )
        if row["sample_std"] is not None:
            rate += f" ± {row['sample_std'] * 100:.1f} pp"
        params = "pending" if row["parameters"] is None else f"{row['parameters']:,}"
        lines.append(f"| {row['label']} | {counts} | {rate} | {params} |")
    lines += [
        "",
        "All learned models use the same 32 training episodes (384 original transitions), with eight held-out validation episodes.",
        "The task list is reused across seeds: these are not 60 independent tasks. No test-score checkpoint selection.",
        "Dreamer uses hindsight goal/reward supervision and the author actor-critic code; it is an offline adaptation.",
        "Dreamer V1/V2 use six-channel images upscaled to 64×64; V3 uses current and goal 32×32 inputs.",
        "Parameter counts describe the adapted networks; target/value copies and architecture differences prevent a parameter-matched algorithm ranking.",
        "Recorded CPU wall times were measured during concurrent jobs and are not valid comparative speed measurements.",
        "Raw task actions, training configurations, metrics, and source revisions are included for audit.",
        "",
    ]
    if missing:
        lines += ["Missing runs: " + ", ".join(missing), ""]
    else:
        lines += [
            "![Success rates and individual seeds](success_rates.png)",
            "",
            "![Per-task outcomes across seeds](task_matrix.png)",
            "",
        ]
    (args.reports / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"completed_reports": sum(r["completed_seeds"] for r in summary), "missing": missing}
        )
    )


if __name__ == "__main__":
    main()
