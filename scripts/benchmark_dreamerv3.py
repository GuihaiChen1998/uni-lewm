"""Author DreamerV3 agent, isolated JAX env, offline goal-conditioned toy data."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from dreamer_data import ROOT, GoalData, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--run-root", type=Path, default=ROOT / "work/benchmark20")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/benchmarks")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import elements
    import jax
    import ruamel.yaml
    from dreamerv3.agent import Agent
    from embodied.jax import internal

    run = args.run_root / f"dreamerv3-seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    data = GoalData(args.seed)
    configs = ruamel.yaml.YAML(typ="safe").load(
        (args.source / "dreamerv3/configs.yaml").read_text()
    )
    configs["defaults"]["jax"].update(profiler=False, precompile=False, transfer_guard=False)
    cfg = (
        elements.Config(configs["defaults"])
        .update(configs["size1m"])
        .update(
            {
                "seed": args.seed,
                "logdir": str(run),
                "batch_size": 16,
                "batch_length": 6,
                "report_length": 6,
                "replay_context": 0,
                "jax.platform": "cpu",
                "jax.compute_dtype": "float32",
                "jax.prealloc": False,
                "jax.profiler": False,
                "jax.precompile": False,
                "jax.transfer_guard": False,
                "agent.enc.simple.mults": [2, 3, 4],
                "agent.dec.simple.mults": [2, 3, 4],
                "agent.dyn.rssm.deter": 128,
                "agent.dyn.rssm.stoch": 4,
                "agent.dyn.rssm.blocks": 4,
                "agent.imag_length": 5,
                "agent.opt.warmup": 100,
                "agent.report": False,
            }
        )
    )
    cfg = elements.Config(
        **cfg.agent,
        logdir=cfg.logdir,
        seed=cfg.seed,
        jax=cfg.jax,
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        replay_context=cfg.replay_context,
        report_length=cfg.report_length,
        replica=0,
        replicas=1,
    )
    (run / "config.json").write_text(json.dumps(dict(cfg), indent=2) + "\n")
    obs = {
        "image": elements.Space(np.uint8, (32, 32, 3)),
        "goal": elements.Space(np.uint8, (32, 32, 3)),
        "reward": elements.Space(np.float32),
        **{k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")},
    }
    agent = Agent(obs, {"action": elements.Space(np.int32, (), 0, 4)}, cfg)
    checkpoint = run / "last.pkl"
    started = time.perf_counter()
    logs = []
    if checkpoint.exists():
        with checkpoint.open("rb") as file:
            agent.load(pickle.load(file))
    else:
        carry = agent.init_train(16)
        for step in range(1, args.steps + 1):
            batch = data.next()
            batch.update(consec=np.zeros((16, 6), np.int32), stepid=np.zeros((16, 6, 20), np.uint8))
            batch = internal.device_put(batch, agent.train_sharded)
            batch["seed"] = agent._seeds(step, agent.train_mirrored)
            # Every batch begins a new independent subsequence (is_first=True).
            carry, _outs, metrics = agent.train(carry, batch)
            if step == 1 or step % 100 == 0 or step == args.steps:
                jax.block_until_ready(agent.params)
                scalars = {k: float(v) for k, v in metrics.items() if np.asarray(v).shape == ()}
                row = {
                    "step": step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics": scalars,
                }
                logs.append(row)
                (run / "metrics.json").write_text(json.dumps(logs, indent=2) + "\n")
                print(json.dumps(row), flush=True)
        with checkpoint.open("wb") as file:
            pickle.dump(agent.save(), file)
    summary = dict(
        **data.provenance(),
        method="dreamerv3",
        seed=args.seed,
        steps=args.steps,
        upstream_commit="e01491fad6434b2245a3b8ca201dd7faedcc458c",
        training_seconds=time.perf_counter() - started,
        parameters=int(
            sum(
                np.prod(x.shape)
                for k, x in agent.params.items()
                if not k.startswith(("opt/", "slowval/"))
            )
        ),
    )
    (run / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    state = None

    def policy(image, goal, first):
        nonlocal state
        if first:
            state = agent.init_policy(1)
        obs = {
            "image": image[None],
            "goal": goal[None],
            "reward": np.zeros(1, np.float32),
            "is_first": np.array([first]),
            "is_last": np.zeros(1, bool),
            "is_terminal": np.zeros(1, bool),
        }
        state, acts, _ = agent.policy(state, obs, mode="eval")
        return int(acts["action"][0])

    # Flush pending parameter synchronization before timing/testing.
    batch = data.next()
    policy(batch["image"][0, 0], batch["goal"][0, 0], True)
    evaluate(
        policy,
        data,
        "dreamerv3",
        args.seed,
        args.report_root / f"dreamerv3-seed{args.seed}.json",
        checkpoint,
    )


if __name__ == "__main__":
    main()
