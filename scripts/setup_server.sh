#!/usr/bin/env bash
# Server setup: venv, dependencies, pipeline check, env autotune hint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON not found. Install Python 3.10 (project requires >=3.10, <3.12)." >&2
  exit 1
fi

echo "[setup] Using $("$PYTHON" --version)"
echo "[setup] Creating venv at $ROOT/.venv"
"$PYTHON" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -e ".[dev,autotune]"

if [[ ! -f "$ROOT/pokemon_hrl/assets/red.gb" ]]; then
  echo "WARN: pokemon_hrl/assets/red.gb missing — place Pokemon Red ROM before training." >&2
fi
if [[ ! -f "$ROOT/pokemon_hrl/assets/pyboy_states/red.state" ]]; then
  echo "WARN: pokemon_hrl/assets/pyboy_states/red.state missing." >&2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

echo "[setup] Running pipeline verification..."
python -m pokemon_hrl.scripts.verify_pipeline

cat <<'EOF'

=== Setup complete ===

1) Env throughput autotune (find num_envs / workers for this machine):
   source .venv/bin/activate
   python -m pokemon_hrl.training.env_autotune --headless

   → Check SPS in output; apply best settings to pokemon_hrl/config/hrl_config.yaml
     (num_envs, num_workers, env_batch_size, vectorization: multiprocessing)

2) PPO hyperparameter search (Optuna):
   python -m pokemon_hrl.training.autotune --headless --n-trials 20

3) Full training:
   python -m pokemon_hrl.training.train_interactive --headless --no-track
   # or with W&B: export WANDB_API_KEY=... && python -m pokemon_hrl.training.train_interactive --headless

Optional env vars:
  WANDB_API_KEY   — Weights & Biases logging
  CUDA_VISIBLE_DEVICES — GPU selection
  MPLCONFIGDIR    — matplotlib cache (default: .matplotlib_cache)

EOF
