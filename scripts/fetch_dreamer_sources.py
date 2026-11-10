"""Fetch author sources at the exact revisions used in the toy comparison."""

import argparse
import subprocess
from pathlib import Path

SOURCES = {
    "dreamer-v1": (
        "https://github.com/danijar/dreamer.git",
        "56d4d444dfd0582b0e79dab80aebbea74c0ce40d",
    ),
    "dreamer-v2": (
        "https://github.com/danijar/dreamerv2.git",
        "07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41",
    ),
    "dreamerv3": (
        "https://github.com/danijar/dreamerv3.git",
        "e01491fad6434b2245a3b8ca201dd7faedcc458c",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/upstream"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in SOURCES.items():
        path = args.output / name
        if path.exists():
            head = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if head != revision:
                raise RuntimeError(
                    f"Existing {path} has a different revision; choose a new output directory"
                )
        else:
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(path)], check=True
            )
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
        print(name, revision)


if __name__ == "__main__":
    main()
