import argparse
import json
from pathlib import Path

from .data import generate_dataset
from .evaluation import evaluate
from .training import train


def main():
    parser = argparse.ArgumentParser(description="Shared-backbone latent world-model prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("generate")
    data.add_argument("--output", required=True)
    data.add_argument("--episodes", type=int, default=256)
    data.add_argument("--steps", type=int, default=48)
    data.add_argument("--image-size", type=int, default=32)
    data.add_argument("--grid-size", type=int, default=9)
    data.add_argument("--seed", type=int, default=42)
    training = sub.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="auto")
    training.add_argument("--resume")
    training.add_argument("--steps", type=int)
    training.add_argument("--seed", type=int)
    training.add_argument("--data")
    training.add_argument("--variant", choices=["shared", "independent", "matched"])
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--max-steps", type=int, default=60)
    evaluation.add_argument("--seed", type=int, default=3001)
    evaluation.add_argument("--stop-threshold", type=float)
    evaluation.add_argument("--random-baseline", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_dataset(
            args.output, args.episodes, args.steps, args.image_size, args.grid_size, args.seed
        )
    elif args.command == "train":
        config = json.loads(Path(args.config).read_text())
        if args.steps is not None:
            config["training"]["steps"] = args.steps
        if args.seed is not None:
            config["training"]["seed"] = args.seed
        if args.data is not None:
            config["data"]["path"] = args.data
        if args.variant:
            config["model"]["shared"] = args.variant != "independent"
            config["parameter_matched"] = args.variant == "matched"
        result = train(config, args.output, args.device, args.resume)
    else:
        result = evaluate(
            args.checkpoint,
            args.output,
            args.episodes,
            args.max_steps,
            args.device,
            args.seed,
            stop_threshold=args.stop_threshold,
            random_baseline=args.random_baseline,
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("results", "train_episode_ids", "validation_episode_ids")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
