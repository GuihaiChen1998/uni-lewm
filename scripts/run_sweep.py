"""Run three controls with identical data, seeds and planning budgets."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/a6000.json")
    parser.add_argument("--output", default="work/sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["shared", "independent", "matched"],
        choices=["shared", "independent", "matched"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    collected = []
    for seed in args.seeds:
        for variant in args.variants:
            directory = Path(args.output) / f"{variant}_seed{seed}"
            command = [
                sys.executable,
                "-m",
                "fused_lewm.cli",
                "train",
                "--config",
                args.config,
                "--output",
                str(directory),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--variant",
                variant,
            ]
            if args.steps is not None:
                command.extend(["--steps", str(args.steps)])
            evaluation = [
                sys.executable,
                "-m",
                "fused_lewm.cli",
                "evaluate",
                "--checkpoint",
                str(directory / "last.pt"),
                "--output",
                str(directory / "evaluation.json"),
                "--device",
                args.device,
                "--episodes",
                str(args.episodes),
                "--max-steps",
                str(args.max_steps),
            ]
            for invocation in (command, evaluation):
                print(shlex.join(invocation), flush=True)
                if not args.dry_run:
                    subprocess.run(invocation, check=True)
            if not args.dry_run:
                summary = json.loads((directory / "summary.json").read_text())
                evaluation = json.loads((directory / "evaluation.json").read_text())
                collected.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "parameters": summary["parameters"],
                        "validation_prediction": summary["validation"]["prediction"],
                        "success_rate": evaluation["success_rate"],
                        "false_stop_rate": evaluation["false_stop_rate"],
                        "mean_planning_seconds": evaluation["mean_planning_seconds"],
                    }
                )
    if not args.dry_run:
        aggregate = {}
        for variant in args.variants:
            rows = [r for r in collected if r["variant"] == variant]
            aggregate[variant] = {}
            for key in (
                "parameters",
                "validation_prediction",
                "success_rate",
                "false_stop_rate",
                "mean_planning_seconds",
            ):
                values = [r[key] for r in rows if r[key] is not None]
                aggregate[variant][key] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std_across_seeds": float(np.std(values)) if values else None,
                }
        report = {
            "runs": collected,
            "aggregate": aggregate,
            "note": "Toy-grid experiment; not an official LeWM benchmark reproduction.",
        }
        (Path(args.output) / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
