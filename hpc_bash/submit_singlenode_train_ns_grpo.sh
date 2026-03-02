#!/bin/bash
#SBATCH --job-name=thetaevolve_train
#SBATCH --output=logs/thetaevolve_train_%j.out
#SBATCH --error=logs/thetaevolve_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G

set -euo pipefail

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Load required modules if the cluster provides module command.
if command -v module >/dev/null 2>&1; then
  module load CUDA/12.4.0 || true
  module load GCC/11.3.0 || true
  module load WebProxy || true
  module load git-lfs || true
fi

# Activate environment from build_conda.sh (preferred: micromamba env "slime").
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate slime
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate slime
else
  echo "ERROR: micromamba/conda not found. Please create env via build_conda.sh first."
  exit 1
fi

# Optional proxy auto-detection for TAMU centers.
NODE_IP="$(hostname -I | awk '{print $1}')"
if [[ "$NODE_IP" == 10.72.* ]]; then
  export http_proxy="${http_proxy:-http://10.72.8.25:8080}"
  export https_proxy="${https_proxy:-http://10.72.8.25:8080}"
elif [[ "$NODE_IP" == 10.73.* ]]; then
  export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
  export https_proxy="${https_proxy:-http://10.73.132.63:8080}"
fi

# Caches and temp directories on scratch.
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/user/$USER}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_BASE/.uv_cache}"
export HF_HOME="${HF_HOME:-$SCRATCH_BASE/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH_BASE/.cache/wandb}"
export WANDB_DIR="${WANDB_DIR:-$WANDB_CACHE_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_BASE/.cache/triton}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR"

# Optional authentication: export tokens before sbatch submit.
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  hf auth login --token "$HUGGINGFACE_HUB_TOKEN"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "$WANDB_API_KEY"
fi

echo "Starting ThetaEvolve training at: $(date)"
echo "Node: $(hostname), JobID: ${SLURM_JOB_ID:-N/A}"
echo "Running command: bash run.sh"
bash run.sh
echo "ThetaEvolve training completed at: $(date)"

