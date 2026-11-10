"""Framework-independent, hindsight goal batches from the SAME train episodes.

No test goals, positions, shortest paths, or environment simulator enter training.
Actions here LEAVE each observation. V1/V2 adapters shift these by one position.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/toy_two_room/smoke.npz"
TASKS = ROOT / "data/toy_two_room/tasks20.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_class():
    # Avoid importing PyTorch in isolated JAX/TensorFlow environments.
    spec = importlib.util.spec_from_file_location("toy_grid", ROOT / "src/fused_lewm/grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TwoRoomGrid


class GoalData:
    def __init__(self, seed, size=32, batch=16, length=6):
        self.rng = np.random.default_rng(seed + 1)
        self.size, self.batch, self.length = size, batch, length
        with np.load(DATA, allow_pickle=False) as data:
            self.images = data["observations"].transpose(0, 1, 3, 4, 2).copy()
            self.actions = data["actions"].copy()
        order = np.random.default_rng(2026).permutation(len(self.images))
        self.validation_ids, self.train_ids = order[:8], order[8:]
        self.tasks = json.loads(TASKS.read_text())

    def resize(self, image):
        if self.size == 32:
            return image
        if self.size == 64:
            return image.repeat(2, axis=-3).repeat(2, axis=-2)
        raise ValueError("only deterministic 32 or 64 pixel inputs supported")

    def next(self):
        images, goals, actions, rewards = [], [], [], []
        for _ in range(self.batch):
            episode = int(self.rng.choice(self.train_ids))
            start = int(self.rng.integers(self.images.shape[1] - self.length + 1))
            end = start + self.length
            obs = self.images[episode, start:end]
            # Sample a future observed goal; no synthetic expert action labels.
            goal = obs[int(self.rng.integers(1, self.length))]
            images.append(obs)
            goals.append(np.broadcast_to(goal, obs.shape))
            act = self.actions[episode, start : min(end, self.actions.shape[1])]
            actions.append(np.pad(act, (0, self.length - len(act))))
            rewards.append(np.all(obs == goal, axis=(1, 2, 3)).astype(np.float32))
        first = np.zeros((self.batch, self.length), bool)
        first[:, 0] = True
        # Goal is a continuing reward state; stopping during evaluation is external.
        return {
            "image": self.resize(np.array(images)),
            "goal": self.resize(np.array(goals)),
            "action": np.array(actions, np.int32),
            "reward": np.array(rewards),
            "is_first": first,
            "is_last": np.zeros_like(first),
            "is_terminal": np.zeros_like(first),
        }

    def provenance(self):
        return {
            "dataset_sha256": digest(DATA),
            "tasks_sha256": digest(TASKS),
            "train_episode_ids": self.train_ids.tolist(),
            "validation_episode_ids": self.validation_ids.tolist(),
            "goal_sampling": "uniform future frame within train-only window",
            "reward": "pixel equality with hindsight goal; continuing task",
            "batch_size": self.batch,
            "batch_length": self.length,
            "image_size": self.size,
            "protocol": "offline goal-conditioned adaptation, not standard online Dreamer",
        }


def evaluate(policy, dataset, method, seed, output, checkpoint):
    """policy(image, goal, is_first)->int. Reset policy state at each task."""
    import time

    env = grid_class()(9, 32, 3001)
    rows, times = [], []
    for index, task in enumerate(dataset.tasks["tasks"]):
        start, goal = tuple(task["start"]), tuple(task["goal"])
        obs = env.reset(start)
        target = env.render(goal)
        actions = []
        reached = False
        for t in range(dataset.tasks["max_steps"] + 1):
            # Pixel equality is the exact same observation information used by
            # JEPA goal encoding, with no privileged coordinate given to policy.
            if np.array_equal(obs, target):
                reached = True
                break
            if t == dataset.tasks["max_steps"]:
                break
            tick = time.perf_counter()
            action = policy(
                dataset.resize(obs.transpose(1, 2, 0)),
                dataset.resize(target.transpose(1, 2, 0)),
                t == 0,
            )
            times.append(time.perf_counter() - tick)
            if not 0 <= action < 4:
                raise ValueError(f"invalid physical action: {action}")
            actions.append(action)
            obs = env.step(action)
        rows.append(
            {
                "episode": index,
                "start": start,
                "goal": goal,
                "reached": reached,
                "stopped": reached,
                "false_stop": False,
                "first_hit_step": len(actions) if reached else None,
                "steps": len(actions),
                "actions": actions,
                "optimal_steps": env.shortest_distance(start, goal),
            }
        )
    report = dict(
        method=method,
        training_seed=seed,
        **dataset.provenance(),
        controller="official actor",
        stop_rule="observed pixel equality",
        episodes=len(rows),
        success_count=sum(r["reached"] for r in rows),
        success_rate=sum(r["reached"] for r in rows) / len(rows),
        false_stop_count=0,
        mean_control_seconds=float(np.mean(times)),
        checkpoint_sha256=digest(checkpoint),
        results=rows,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    return report
