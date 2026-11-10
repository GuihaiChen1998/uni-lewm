"""Author Dreamer V1/V2 agents with TF2.15 compatibility and goal image input.

Losses and imagination/actor/value training come from pinned author sources.
Compatibility shims are API renames, not alternate learning algorithms.
"""

import argparse
import collections
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
from dreamer_data import ROOT, GoalData, evaluate


def prepare_batch(data):
    result = data.copy()
    result["image"] = np.concatenate([result["image"], result.pop("goal")], -1)
    onehot = np.eye(4, dtype=np.float32)[result["action"]]
    # V1/V2 replay action[t] ENTERS observation[t]. V3 uses outgoing actions.
    result["action"] = np.concatenate([np.zeros_like(onehot[:, :1]), onehot[:, :-1]], 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=[1, 2], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--run-root", type=Path, default=ROOT / "work/benchmark20")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/benchmarks")
    args = parser.parse_args()
    import gym
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    # Removed Keras 'experimental' namespace. Use the public float32 APIs.
    public = tf.keras.mixed_precision
    shim = types.SimpleNamespace(
        global_policy=public.global_policy,
        set_policy=public.set_global_policy,
        Policy=public.Policy,
        LossScaleOptimizer=lambda opt, scale: public.LossScaleOptimizer(opt, dynamic=True),
    )
    tf.keras.mixed_precision.experimental = shim
    importlib.import_module("tensorflow.keras.mixed_precision").experimental = shim
    public.set_global_policy("float32")
    # Keras 2.15 optimizers changed lazy variable ownership; preserve legacy Adam.
    tf.optimizers.Adam = tf.keras.optimizers.legacy.Adam
    data = GoalData(args.seed, size=64)
    method = f"dreamerv{args.version}"
    run = args.run_root / f"{method}-seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "last.pkl"
    if checkpoint.exists():
        raise FileExistsError(f"Use a fresh run directory: {run}")
    logs = []
    started = time.perf_counter()
    if args.version == 2:
        sys.path.insert(0, str(args.source.resolve() / "dreamerv2"))
        import agent as upstream
        import common
        import ruamel.yaml

        defaults = ruamel.yaml.safe_load((args.source / "dreamerv2/configs.yaml").read_text())[
            "defaults"
        ]
        cfg = common.Config(defaults).update(
            {
                "seed": args.seed,
                "logdir": str(run),
                "precision": 32,
                "jit": False,
                "dataset.batch": 16,
                "dataset.length": 6,
                "rssm.deter": 128,
                "rssm.hidden": 64,
                "rssm.stoch": 4,
                "rssm.discrete": 4,
                "encoder.cnn_depth": 4,
                "decoder.cnn_depth": 4,
                "encoder.mlp_keys": "$^",
                "decoder.mlp_keys": "$^",
                "actor.layers": 2,
                "actor.units": 64,
                "actor.dist": "onehot",
                "critic.layers": 2,
                "critic.units": 64,
                "actor_grad": "reinforce",
                "reward_head.layers": 2,
                "reward_head.units": 64,
                "discount_head.layers": 2,
                "discount_head.units": 64,
                "imag_horizon": 5,
                "clip_rewards": "identity",
            }
        )
        obs_space = {
            "image": gym.spaces.Box(0, 255, (64, 64, 6), np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), np.float32),
            **{k: gym.spaces.Box(0, 1, (), bool) for k in ("is_first", "is_last", "is_terminal")},
        }
        act_space = {"action": gym.spaces.Box(0, 1, (4,), np.float32)}
        model = upstream.Agent(cfg, obs_space, act_space, common.Counter())

        def train_step(batch):
            _, metrics = model.train(prepare_batch(batch), None)
            return {k: float(v) for k, v in metrics.items()}

        state = None

        def policy(image, goal, first):
            nonlocal state
            if first:
                state = None
            obs = {
                "image": np.concatenate([image, goal], -1)[None],
                "reward": np.zeros(1, np.float32),
                "is_first": np.array([first]),
                "is_last": np.zeros(1, bool),
                "is_terminal": np.zeros(1, bool),
            }
            acts, state = model.policy(obs, state, mode="eval")
            return int(np.asarray(acts["action"])[0].argmax())

        first_step = 1
        config_dict = dict(cfg)
        commit = "07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41"
    else:
        sys.path.insert(0, str(args.source.resolve()))
        import dreamer as upstream
        import models

        original_decoder = models.ConvDecoder

        class GoalDecoder(original_decoder):
            def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 6)):
                super().__init__(depth, act, shape)

        models.ConvDecoder = GoalDecoder
        cfg = upstream.define_config()
        cfg.update(
            seed=args.seed,
            logdir=run,
            precision=32,
            batch_size=16,
            batch_length=6,
            deter_size=128,
            stoch_size=16,
            num_units=64,
            cnn_depth=4,
            horizon=5,
            action_dist="onehot",
            expl="epsilon_greedy",
            expl_amount=0.0,
            pcont=True,
            log_images=False,
            log_scalars=True,
            action_repeat=1,
        )
        first_batch = data.next()

        class OfflineDreamer(upstream.Dreamer):
            def __init__(self):
                self._c = cfg
                self._actdim = 4
                self._actspace = gym.spaces.Discrete(4)
                self._float = "float32"
                self._writer = tf.summary.create_noop_writer()
                self._step = tf.Variable(0, dtype=tf.int64)
                self._metrics = collections.defaultdict(tf.metrics.Mean)
                self._strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
                batch = prepare_batch(first_batch)
                batch["discount"] = 1 - batch["is_terminal"].astype(np.float32)
                self._dataset = iter([upstream.preprocess(batch, cfg)])
                with self._strategy.scope():
                    self._build_model()  # includes exactly one optimizer update

            @tf.function
            def train(self, batch, log_images=False):
                self._strategy.run(self._train, args=(batch, log_images))

        model = OfflineDreamer()

        def train_step(batch):
            batch = prepare_batch(batch)
            batch["discount"] = 1 - batch["is_terminal"].astype(np.float32)
            model.train(upstream.preprocess(batch, cfg))
            return {k: float(v.result()) for k, v in model._metrics.items()}

        state = None

        def policy(image, goal, first):
            nonlocal state
            if first:
                state = None
            obs = {
                "image": np.concatenate([image, goal], -1)[None],
                "reward": np.zeros(1, np.float32),
            }
            acts, state = model.policy(obs, state, False)
            return int(np.asarray(acts)[0].argmax())

        first_step = 2
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
        commit = "56d4d444dfd0582b0e79dab80aebbea74c0ce40d"
    (run / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    for step in range(first_step, args.steps + 1):
        metrics = train_step(data.next())
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"nonfinite training metric at {step}")
        if step == first_step or step % 100 == 0 or step == args.steps:
            row = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": metrics,
            }
            logs.append(row)
            (run / "metrics.json").write_text(json.dumps(logs, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    model.save(checkpoint)
    summary = dict(
        **data.provenance(),
        method=method,
        seed=args.seed,
        steps=args.steps,
        upstream_commit=commit,
        training_seconds=time.perf_counter() - started,
        parameters=int(sum(np.prod(x.shape) for x in model.trainable_variables)),
    )
    (run / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    batch = data.next()
    policy(batch["image"][0, 0], batch["goal"][0, 0], True)
    evaluate(
        policy,
        data,
        method,
        args.seed,
        args.report_root / f"{method}-seed{args.seed}.json",
        checkpoint,
    )


if __name__ == "__main__":
    main()
