import json
from pathlib import Path

import numpy as np
import torch

from .data import file_sha256
from .grid import TwoRoomGrid
from .planner import CEMPlanner, PlannerConfig
from .training import load_model, select_device


def image_tensor(image, device):
    return torch.from_numpy(image.copy()).float().div_(255).unsqueeze(0).to(device)


def make_tasks(env, episodes, seed=3001):
    """Legacy evaluator task generation; seed 3001 matches committed tasks20.json."""
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(episodes):
        indices = rng.choice(len(env.cells), size=2, replace=False)
        tasks.append((env.cells[int(indices[0])], env.cells[int(indices[1])]))
    return tasks


@torch.inference_mode()
def calibrate_grid_stop(model, env, validation_cells):
    """Use held-out trajectory positions only; test episodes are not examined.

    This deterministic grid has identical frames on revisits. For real-camera
    tasks, replace positives with independent repeated/noisy observations and
    calibrate against environment-specific goal-equivalence labels.
    """
    device = next(model.parameters()).device
    cells = sorted({tuple(map(int, p)) for p in validation_cells})
    if len(cells) < 2:
        raise ValueError("need at least two distinct validation positions to calibrate STOP")
    images = torch.stack([image_tensor(env.render(cell), device)[0] for cell in cells])
    z = model.encode(images)
    distances = (z[:, None] - z[None, :]).square().mean(-1)
    mask = ~torch.eye(len(cells), dtype=torch.bool, device=device)
    negative = distances[mask]
    if not torch.isfinite(negative).all() or float(negative.min()) <= 1e-12:
        raise ValueError("degenerate goal representation; STOP calibration failed")
    threshold = float(negative.min()) * 0.5
    return threshold, {
        "method": "half_min_validation_negative_distance",
        "validation_cells": len(cells),
        "positive_distance": 0.0,
        "negative_min_distance": float(negative.min()),
        "calibration_false_positive_rate": float((negative <= threshold).float().mean()),
    }


@torch.inference_mode()
def evaluate(
    checkpoint,
    output,
    episodes=20,
    max_steps=60,
    device_name="auto",
    seed=3001,
    planner_config=None,
    stop_threshold=None,
    random_baseline=False,
):
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    torch.set_num_threads(payload["config"]["training"].get("cpu_threads", 2))
    data_cfg = payload["config"]["data"]
    dataset_path = data_cfg["path"]
    if file_sha256(dataset_path) != payload["dataset_sha256"]:
        raise ValueError("evaluation dataset fingerprint differs from training")
    with np.load(dataset_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("environment") != "toy_two_room_grid":
            raise ValueError("this evaluator is only for the bundled toy grid")
        order = np.random.default_rng(data_cfg["split_seed"]).permutation(
            len(archive["observations"])
        )
        n_val = max(1, min(len(order) - 1, round(len(order) * 0.2)))
        validation_positions = archive["positions"][order[:n_val]].reshape(-1, 2)
    env = TwoRoomGrid(metadata["grid_size"], model.config.image_size, seed)
    calibration = {"method": "explicit_threshold"}
    if stop_threshold is None:
        stop_threshold, calibration = calibrate_grid_stop(model, env, validation_positions)
    cfg = PlannerConfig(**(planner_config or payload["config"].get("planner", {})))
    planner = CEMPlanner(model, cfg, stop_threshold, seed=seed + 1)
    # Separate streams keep test tasks fixed across different controllers.
    action_rng = np.random.default_rng(seed + 2)
    tasks = make_tasks(env, episodes, seed)
    rows, all_times = [], []
    for index, (start, goal_position) in enumerate(tasks):
        obs = env.reset(start)
        goal = model.encode_goal(image_tensor(env.render(goal_position), device)).state
        history = model.encode(image_tensor(obs, device))[:, None]
        past = torch.empty(1, 0, dtype=torch.long, device=device)
        reached, stopped, false_stop, first_hit, actions_taken = False, False, False, None, []
        for step in range(max_steps + 1):
            at_goal = env.position == goal_position  # evaluator only, never passed to planner
            if at_goal:
                reached = True
                if first_hit is None:
                    first_hit = step
            if step == max_steps:
                break
            if random_baseline:
                stopped = float(planner.goal_distance(history[:, -1], goal)) <= stop_threshold
                action = model.stop_index if stopped else int(action_rng.integers(env.num_actions))
            else:
                plan = planner.plan(history, past, goal)
                action, stopped = plan.action, plan.stopped
                all_times.append(plan.seconds)
            if stopped:
                false_stop = not at_goal
                break
            actions_taken.append(action)
            next_obs = env.step(action)
            next_z = model.encode(image_tensor(next_obs, device))[:, None]
            history = torch.cat([history, next_z], dim=1)[:, -model.config.history :]
            past = torch.cat([past, torch.tensor([[action]], device=device)], dim=1)
            past = past[:, -(history.shape[1] - 1) :] if history.shape[1] > 1 else past[:, :0]
        row = {
            "episode": index,
            "start": start,
            "goal": goal_position,
            "reached": reached,
            "stopped": stopped,
            "false_stop": false_stop,
            "first_hit_step": first_hit,
            "steps": len(actions_taken),
            "optimal_steps": env.shortest_distance(start, goal_position),
            "actions": actions_taken,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "actions"}), flush=True)
    report = {
        "environment": "toy_two_room_grid (not official TwoRoom)",
        "seed": seed,
        "checkpoint_sha256": file_sha256(checkpoint),
        "episodes": episodes,
        "controller": "random" if random_baseline else "categorical_cem",
        "success_rate": float(np.mean([r["reached"] for r in rows])),
        "false_stop_rate": float(np.mean([r["false_stop"] for r in rows])),
        "stop_threshold": stop_threshold,
        "calibration": calibration,
        "mean_planning_seconds": float(np.mean(all_times)) if all_times else None,
        "parameters": model.parameter_count(),
        "results": rows,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
