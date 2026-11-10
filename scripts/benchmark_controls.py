"""Random policy and privileged shortest-path upper bound on the fixed task file."""

import json
from pathlib import Path

import numpy as np
from dreamer_data import DATA, TASKS, digest, grid_class


def main():
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads(TASKS.read_text())
    for controller in ("random", "oracle_shortest_path"):
        for seed in range(3 if controller == "random" else 1):
            env = grid_class()(9, 32, 3001)
            rng = np.random.default_rng(3003 + seed)
            rows = []
            for index, task in enumerate(tasks["tasks"]):
                start, goal = tuple(task["start"]), tuple(task["goal"])
                obs = env.reset(start)
                target = env.render(goal)
                actions = []
                reached = False
                for _ in range(tasks["max_steps"] + 1):
                    if np.array_equal(obs, target):
                        reached = True
                        break
                    if len(actions) == tasks["max_steps"]:
                        break
                    if controller == "random":
                        action = int(rng.integers(4))
                    else:
                        position = env.position
                        costs = []
                        for a in range(4):
                            env.reset(position)
                            env.step(a)
                            costs.append(env.shortest_distance(env.position, goal))
                        env.reset(position)
                        action = int(np.argmin(costs))
                    actions.append(action)
                    obs = env.step(action)
                rows.append(
                    {
                        "episode": index,
                        "start": start,
                        "goal": goal,
                        "reached": reached,
                        "steps": len(actions),
                        "actions": actions,
                    }
                )
            report = {
                "method": controller,
                "training_seed": None,
                "evaluation_seed": 3003 + seed,
                "dataset_sha256": digest(DATA),
                "tasks_sha256": digest(TASKS),
                "controller": controller,
                "privileged": controller != "random",
                "success_count": sum(x["reached"] for x in rows),
                "episodes": 20,
                "success_rate": sum(x["reached"] for x in rows) / 20,
                "results": rows,
            }
            (root / f"reports/benchmarks/{controller}-seed{seed}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(controller, seed, report["success_count"])


if __name__ == "__main__":
    main()
