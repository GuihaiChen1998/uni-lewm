#!/usr/bin/env python3
"""Train named toy adaptations and evaluate the committed, immutable task list."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fused_lewm.baselines import ToyBaseline
from fused_lewm.data import TrajectoryDataset, file_sha256
from fused_lewm.evaluation import calibrate_grid_stop, image_tensor
from fused_lewm.grid import TwoRoomGrid
from fused_lewm.losses import SIGReg, world_model_loss
from fused_lewm.model import ModelConfig, WorldModel
from fused_lewm.planner import CEMPlanner, PlannerConfig
from fused_lewm.training import atomic_save, batch_from, seed_everything

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/toy_two_room/smoke.npz"
TASKS = ROOT / "data/toy_two_room/tasks20.json"
PLANNER = PlannerConfig(horizon=4, candidates=24, elites=6, iterations=2, chunk_size=24)


def make_model(method):
    if method == "uni_lewm":
        return WorldModel(
            ModelConfig(
                image_size=32,
                patch_size=8,
                dim=48,
                depth=2,
                heads=3,
                mlp_ratio=2,
                history=2,
                num_actions=4,
                shared=True,
                projection_batchnorm=True,
            )
        )
    return ToyBaseline(method)


def train_one(method, seed, steps, output, device):
    seed_everything(seed)
    model = make_model(method).to(device)
    train = TrajectoryDataset(DATA, 2, 4, "train", 2026)
    validation = TrajectoryDataset(DATA, 2, 4, "validation", 2026)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    reg = SIGReg(32).to(device)
    config = {
        "method": method,
        "seed": seed,
        "steps": steps,
        "batch_size": 16,
        "history": 2,
        "horizon": 4,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1,
        "sigreg_weight_normalized": 1,
        "sigreg_projections": 32,
        "dataset_sha256": train.fingerprint,
        "split_seed": 2026,
        "train_episode_ids": train.episode_ids.tolist(),
        "validation_episode_ids": validation.episode_ids.tolist(),
        "dim": 48,
        "depth": 2,
        "cpu_threads": torch.get_num_threads(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": str(device),
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, weights_only=True, map_location=device)
        if payload["config"] != config or payload["step"] != steps:
            raise ValueError(f"Existing run differs from requested run: {output}")
        model.load_state_dict(payload["model"])
        return model.eval(), validation
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    logs = []
    model.train()
    for step in range(1, steps + 1):
        obs, actions = batch_from(train, 16, generator, device)
        optimizer.zero_grad(set_to_none=True)
        if method == "uni_lewm":
            loss, metrics = world_model_loss(model, obs, actions, reg, 1.0)
        else:
            loss, metrics = model.loss(obs, actions, (step - 1) / steps)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{method}: nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row = {
                "step": step,
                **{k: float(v) for k, v in metrics.items()},
                "elapsed_seconds": time.perf_counter() - start,
            }
            logs.append(row)
            print(json.dumps({"method": method, "seed": seed, **row}), flush=True)
            (output / "metrics.json").write_text(json.dumps(logs, indent=2) + "\n")
    model.eval()
    val_metrics = []
    with (
        torch.no_grad(),
        torch.random.fork_rng(devices=[] if device.type == "cpu" else [device.index or 0]),
    ):
        torch.manual_seed(771)
        vg = torch.Generator().manual_seed(771)
        for _ in range(4):
            obs, actions = batch_from(validation, 16, vg, device)
            if method == "uni_lewm":
                z = model.encode(obs.flatten(0, 1)).reshape(16, 6, -1)
            else:
                z = model.encode_sequence(obs)
            predicted = model.rollout(z[:, :2], actions[:, :1], actions[:, 1:])
            val_metrics.append(
                {
                    "latent_std": float(z.std(0).mean()),
                    "rollout_mse": float((predicted - z[:, 2:]).square().mean()),
                }
            )
    val_metrics = {k: sum(r[k] for r in val_metrics) / len(val_metrics) for k in val_metrics[0]}
    summary = {
        "config": config,
        "parameters": model.parameter_count(),
        "validation": val_metrics,
        "training_seconds": time.perf_counter() - start,
        "collapsed": val_metrics["latent_std"] < 0.05,
    }
    (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    atomic_save(
        {
            "config": config,
            "model": model.state_dict(),
            "step": steps,
            "optimizer": optimizer.state_dict(),
            "data_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        checkpoint,
    )
    return model, validation


@torch.inference_mode()
def evaluate(model, validation, method, seed, checkpoint, output, direct=False):
    model.eval()
    device = next(model.parameters()).device
    tasks = json.loads(TASKS.read_text())
    env = TwoRoomGrid(9, 32, 3001)
    with np.load(DATA, allow_pickle=False) as data:
        positions = data["positions"][validation.episode_ids].reshape(-1, 2)
    threshold, calibration = calibrate_grid_stop(model, env, positions)
    planner = CEMPlanner(model, PLANNER, threshold, seed=3002)
    rows, times = [], []
    for index, task in enumerate(tasks["tasks"]):
        start, goal = tuple(task["start"]), tuple(task["goal"])
        obs = env.reset(start)
        zg = model.encode(image_tensor(env.render(goal), device))
        history = model.encode(image_tensor(obs, device))[:, None]
        past = torch.empty(1, 0, device=device, dtype=torch.long)
        acts, reached, stopped, false_stop, hit = [], False, False, False, None
        for t in range(tasks["max_steps"] + 1):
            at_goal = env.position == goal  # evaluator truth, not given to controller
            if at_goal:
                reached = True
                if hit is None:
                    hit = t
            if t == tasks["max_steps"]:
                break
            tick = time.perf_counter()
            if direct:
                stopped = bool(planner.goal_distance(history[:, -1], zg) <= threshold)
                action = model.stop_index if stopped else model.direct_action(history, past, zg)
            else:
                plan = planner.plan(history, past, zg)
                stopped, action = plan.stopped, plan.action
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - tick)
            if stopped:
                false_stop = not at_goal
                break
            acts.append(action)
            obs = env.step(action)
            history = torch.cat([history, model.encode(image_tensor(obs, device))[:, None]], 1)[
                :, -2:
            ]
            past = torch.cat([past, torch.tensor([[action]], device=device)], 1)[:, -1:]
        rows.append(
            {
                "episode": index,
                "start": start,
                "goal": goal,
                "reached": reached,
                "stopped": stopped,
                "false_stop": false_stop,
                "first_hit_step": hit,
                "steps": len(acts),
                "actions": acts,
                "optimal_steps": env.shortest_distance(start, goal),
            }
        )
    result = {
        "method": method,
        "training_seed": seed,
        "implementation": "toy-scale adaptation; not paper benchmark",
        "controller": "discretized_gaussian_actor" if direct else "categorical_cem",
        "dataset_sha256": file_sha256(DATA),
        "tasks_sha256": file_sha256(TASKS),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameters": model.parameter_count(),
        "task_seed": 3001,
        "planner_seed": 3002,
        "planner": asdict(PLANNER) if not direct else None,
        "success_count": sum(x["reached"] for x in rows),
        "episodes": len(rows),
        "success_rate": sum(x["reached"] for x in rows) / len(rows),
        "false_stop_count": sum(x["false_stop"] for x in rows),
        "mean_control_seconds": float(np.mean(times)),
        "stop_threshold": threshold,
        "calibration": calibration,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+", default=["uni_lewm", "lewm", "sub_lewm", "vlwm", "flexiworld"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--run-root", type=Path, default=ROOT / "work/benchmark20")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/benchmarks")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    for seed in args.seeds:
        for method in args.methods:
            run = args.run_root / f"{method}-seed{seed}"
            model, validation = train_one(method, seed, args.steps, run, device)
            evaluate(
                model,
                validation,
                method,
                seed,
                run / "last.pt",
                args.report_root / f"{method}-seed{seed}.json",
            )
            if method == "flexiworld":
                evaluate(
                    model,
                    validation,
                    method,
                    seed,
                    run / "last.pt",
                    args.report_root / f"{method}-direct-seed{seed}.json",
                    direct=True,
                )


if __name__ == "__main__":
    main()
