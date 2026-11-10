import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import TwoRoomGrid


def generate_dataset(
    path: str | Path,
    episodes: int = 256,
    steps: int = 48,
    image_size: int = 32,
    grid_size: int = 9,
    seed: int = 42,
) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {path}")
    if episodes < 4 or steps < 1:
        raise ValueError("need at least four episodes and one step")
    env = TwoRoomGrid(grid_size, image_size, seed)
    observations = np.empty((episodes, steps + 1, 3, image_size, image_size), dtype=np.uint8)
    actions = np.empty((episodes, steps), dtype=np.int64)
    positions = np.empty((episodes, steps + 1, 2), dtype=np.int16)
    for ep in range(episodes):
        observations[ep, 0] = env.reset()
        positions[ep, 0] = env.position
        for t in range(steps):
            action = int(env.rng.integers(env.num_actions))
            actions[ep, t] = action
            observations[ep, t + 1] = env.step(action)
            positions[ep, t + 1] = env.position
    metadata = {
        "environment": "toy_two_room_grid",
        "version": 1,
        "episodes": episodes,
        "steps": steps,
        "image_size": image_size,
        "grid_size": grid_size,
        "num_actions": 4,
        "seed": seed,
        "policy": "uniform_random",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        positions=positions,
        metadata=json.dumps(metadata),
    )
    return metadata


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TrajectoryDataset(Dataset):
    """Fixed-length NPZ episodes, split by episode before making overlapping windows.

    External discrete-action datasets can use the same schema. A transition must
    never cross a reset; pad-free episodes are required by this initial adapter.
    """

    def __init__(
        self,
        path: str | Path,
        context: int,
        horizon: int,
        split: str,
        split_seed: int = 2026,
        validation_fraction: float = 0.2,
    ):
        if context < 1 or horizon < 1 or not 0 < validation_fraction < 1:
            raise ValueError("invalid context, horizon or validation_fraction")
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            self.observations = archive["observations"]
            self.actions = archive["actions"]
            self.metadata = json.loads(str(archive["metadata"])) if "metadata" in archive else {}
        obs, actions = self.observations, self.actions
        if obs.ndim != 5 or obs.dtype != np.uint8:
            raise ValueError("observations must be uint8 [episodes,T+1,C,H,W]")
        if actions.ndim != 2 or actions.shape != (obs.shape[0], obs.shape[1] - 1):
            raise ValueError("actions must align with observations as [episodes,T]")
        if not np.issubdtype(actions.dtype, np.integer) or (actions < 0).any():
            raise ValueError("actions must be nonnegative discrete IDs")
        if len(obs) < 4:
            raise ValueError("need at least four episodes for an episode-level split")
        self.window = context + horizon
        self.windows_per_episode = obs.shape[1] - self.window + 1
        if self.windows_per_episode < 1:
            raise ValueError("episodes shorter than context+horizon")
        indices = np.random.default_rng(split_seed).permutation(len(obs))
        n_val = max(1, min(len(obs) - 1, round(len(obs) * validation_fraction)))
        self.episode_ids = indices[n_val:] if split == "train" else indices[:n_val]
        self.fingerprint = file_sha256(path)

    def __len__(self):
        return len(self.episode_ids) * self.windows_per_episode

    def __getitem__(self, index):
        ep = self.episode_ids[index // self.windows_per_episode]
        start = index % self.windows_per_episode
        end = start + self.window
        obs = torch.from_numpy(self.observations[ep, start:end].copy()).float().div_(255)
        actions = torch.from_numpy(self.actions[ep, start : end - 1].astype(np.int64, copy=True))
        return obs, actions
