import json
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import TrajectoryDataset
from .losses import SIGReg, world_model_loss
from .model import ModelConfig, WorldModel, matched_independent_config


def select_device(request: str) -> torch.device:
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for smoke tests")
    return device


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_from(dataset, size, generator, device):
    ids = torch.randint(len(dataset), (size,), generator=generator)
    pairs = [dataset[int(index)] for index in ids]
    return tuple(torch.stack(items).to(device) for items in zip(*pairs))


@torch.no_grad()
def validate(model, dataset, regularizer, settings, device):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(settings.get("validation_seed", 771))
    devices = [device.index or 0] if device.type == "cuda" else []
    results = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(settings.get("validation_seed", 771))
        for _ in range(settings.get("validation_batches", 4)):
            obs, actions = batch_from(dataset, settings["batch_size"], generator, device)
            _, metrics = world_model_loss(
                model, obs, actions, regularizer, settings["sigreg_weight"]
            )
            results.append({key: float(value) for key, value in metrics.items()})
    model.train(was_training)
    return {key: sum(result[key] for result in results) / len(results) for key in results[0]}


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_config = dict(payload["config"]["model"])
    model_config.setdefault(
        "projection_batchnorm", False
    )  # pre-normalization prototype checkpoints
    model = WorldModel(ModelConfig(**model_config))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def train(
    config: dict, output: str | Path, device_name: str = "auto", resume: str | Path | None = None
) -> dict:
    config = json.loads(json.dumps(config))
    settings = config["training"]
    for key in ("steps", "batch_size", "horizon", "log_every", "validate_every", "save_every"):
        if settings[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if settings["batch_size"] < 2 or settings["sigreg_weight"] <= 0:
        raise ValueError("need batch_size>=2 and a positive SIGReg weight")
    device = select_device(device_name)
    torch.set_num_threads(settings.get("cpu_threads", 2))
    seed_everything(settings["seed"])
    model_config = ModelConfig(**config["model"])
    if config.get("parameter_matched", False):
        model_config = matched_independent_config(model_config)
    config["model"] = asdict(model_config)
    # Width-search initialization must not change the initialization/data seed.
    seed_everything(settings["seed"])
    model = WorldModel(model_config).to(device)
    regularizer = SIGReg(settings.get("sigreg_projections", 64)).to(device)
    data_settings = config["data"]
    train_data = TrajectoryDataset(
        data_settings["path"],
        model_config.history,
        settings["horizon"],
        "train",
        data_settings["split_seed"],
    )
    val_data = TrajectoryDataset(
        data_settings["path"],
        model_config.history,
        settings["horizon"],
        "validation",
        data_settings["split_seed"],
    )
    if train_data.observations.shape[2:] != (
        model_config.channels,
        model_config.image_size,
        model_config.image_size,
    ):
        raise ValueError("dataset image shape does not match model")
    if train_data.actions.max() >= model_config.num_actions:
        raise ValueError("dataset contains actions outside model's physical action set")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings.get("weight_decay", 1e-4),
    )
    data_generator = torch.Generator().manual_seed(settings["seed"] + 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (
        any((output / name).exists() for name in ("last.pt", "config.json", "metrics.jsonl"))
        and resume is None
    ):
        raise FileExistsError("run exists; use --resume or a new output directory")
    start_step = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["dataset_sha256"] != train_data.fingerprint:
            raise ValueError("resume dataset fingerprint mismatch")
        if saved["config"]["model"] != config["model"] or saved["config"]["data"] != config["data"]:
            raise ValueError("resume model/data configuration mismatch")
        mutable = {"steps", "log_every", "validate_every", "save_every", "cpu_threads"}
        old = {k: v for k, v in saved["config"]["training"].items() if k not in mutable}
        new = {k: v for k, v in settings.items() if k not in mutable}
        if old != new:
            raise ValueError("resume training configuration mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        data_generator.set_state(saved["data_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_step = saved["step"]
    if start_step >= settings["steps"]:
        raise ValueError("requested steps must exceed checkpoint step")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata = {
        "device": str(device),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "parameters": model.parameter_count(),
        "shared": model_config.shared,
        "dataset_sha256": train_data.fingerprint,
        "dataset_metadata": train_data.metadata,
        "train_episode_ids": train_data.episode_ids.tolist(),
        "validation_episode_ids": val_data.episode_ids.tolist(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "implementation": "fused-lewm-prototype-v0.1 (not official LeWM)",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    amp = bool(
        settings.get("amp", True) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    started = time.perf_counter()
    model.train()
    last_validation = None
    for step in range(start_step + 1, settings["steps"] + 1):
        collapsed = False
        observations, actions = batch_from(
            train_data, settings["batch_size"], data_generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss, metrics = world_model_loss(
                model, observations, actions, regularizer, settings["sigreg_weight"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("grad_clip", 1.0))
        if not torch.isfinite(norm):
            raise FloatingPointError(f"non-finite gradients at step {step}")
        optimizer.step()
        final_step = step == settings["steps"]
        if step % settings["log_every"] == 0 or step == 1 or final_step:
            row = {
                "step": step,
                "split": "train",
                **{k: float(v) for k, v in metrics.items()},
                "grad_norm": float(norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % settings["validate_every"] == 0 or final_step:
            last_validation = validate(model, val_data, regularizer, settings, device)
            if not all(np.isfinite(value) for value in last_validation.values()):
                raise FloatingPointError("non-finite validation statistics")
            collapsed = step >= settings.get("collapse_check_after", 500) and last_validation[
                "latent_std"
            ] < settings.get("minimum_latent_std", 0.05)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"step": step, "split": "validation", **last_validation}) + "\n"
                )
        if step % settings["save_every"] == 0 or final_step or collapsed:
            payload = {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "dataset_sha256": train_data.fingerprint,
                "torch_rng": torch.get_rng_state(),
                "data_rng": data_generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            atomic_save(payload, output / "last.pt")
        if collapsed:
            failure = {
                "step": step,
                "reason": "low latent variance / possible representation collapse",
                "validation": last_validation,
            }
            (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise RuntimeError(
                "possible representation collapse; checkpoint and failure.json saved"
            )
    summary = {
        **metadata,
        "step": settings["steps"],
        "validation": last_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
