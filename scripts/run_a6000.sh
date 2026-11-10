#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-}"
if [[ "$mode" != smoke && "$mode" != full ]]; then
  echo "Usage: bash scripts/run_a6000.sh smoke|full" >&2
  exit 2
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("A CUDA-enabled PyTorch environment is required.")
name = torch.cuda.get_device_name(0)
if "RTX A6000" not in name:
    raise SystemExit(f"Expected NVIDIA RTX A6000, found {name}")
print({"gpu": name, "memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
       "torch": str(torch.__version__), "cuda": torch.version.cuda})
PY
# Preserve the CUDA-enabled torch supplied by the selected RunPod image.
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
if [[ "$mode" == smoke ]]; then
  if [[ ! -f work/data/smoke.npz ]]; then
    .venv/bin/fused-lewm generate --output work/data/smoke.npz --episodes 40 --steps 12
  fi
  timeout --signal=INT --kill-after=60s "${MAX_RUN_SECONDS:-1800}" \
    .venv/bin/python scripts/run_sweep.py --config configs/smoke.json \
    --output "${RUN_OUTPUT:-work/a6000_smoke}" --device cuda --seeds 0 --episodes 3 --max-steps 12
else
  if [[ ! -f work/data/two_room_grid.npz ]]; then
    .venv/bin/fused-lewm generate --output work/data/two_room_grid.npz --episodes 1024 --steps 64
  fi
  timeout --signal=INT --kill-after=60s "${MAX_RUN_SECONDS:-21600}" \
    .venv/bin/python scripts/run_sweep.py --config configs/a6000.json \
    --output "${RUN_OUTPUT:-work/a6000_full}" --device cuda --seeds 0 1 2
fi
# This script does not create/stop/delete Pods. Instance shutdown is a separate
# control-plane action after results are retrieved; process exit alone is not it.
