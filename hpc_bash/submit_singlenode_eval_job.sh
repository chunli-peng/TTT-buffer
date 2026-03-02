#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source openr1/bin/activate

# Load any required modules (adjust based on your HPC system)
module load CUDA/12.4.0
module load GCC/11.3.0
module load WebProxy
module load git-lfs

# Set environment variables
export UV_CACHE_DIR=/scratch/user/chunli.peng/.uv_cache
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/scratch/user/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/user/$USER/.cache/wandb
export NODE_IP=$(hostname -i | awk '{print $1}')
export VLLM_HOST_IP=$NODE_IP
export HOST_IP=$NODE_IP

# Set proxy for TAMU HPC Center
if [[ $NODE_IP == 10.72.* ]]; then
    # Faster center (IP range 10.72.x.x)
    echo "Detected Faster center (IP: $NODE_IP)"
    export http_proxy=http://10.72.8.25:8080
    export https_proxy=http://10.72.8.25:8080
elif [[ $NODE_IP == 10.73.* ]]; then
    # Grace center (IP range 10.73.x.x)
    echo "Detected Grace center (IP: $NODE_IP)"
    export http_proxy=http://10.73.132.63:8080
    export https_proxy=http://10.73.132.63:8080
else
    # Unknown computation center - stop execution
    echo "ERROR: Unknown computation center detected!"
    echo "IP Address: $NODE_IP"
    echo "Expected IP ranges: 10.72.x.x (Faster) or 10.73.x.x (Grace)"
    echo "Cannot proceed without proper proxy configuration."
    exit 1
fi

# Require credentials to be provided via environment variables.
# Example: export HUGGINGFACE_HUB_TOKEN=... and export WANDB_API_KEY=...
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: HUGGINGFACE_HUB_TOKEN is not set"
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not set"
    exit 1
fi

# Hugging Face and Weights & Biases authentication
git config --global credential.helper store
hf auth login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential
wandb login --relogin "$WANDB_API_KEY"

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $WANDB_CACHE_DIR

# Run the evaluation
echo "Starting evaluation..."
echo "Timestamp: $(date)"

# Make the evaluation script executable
chmod +x eval_pass@k.sh

# Run the evaluation command
# sh eval_pass@k.sh chunli-peng/DeepSeek-R1-Distill-Qwen-1.5B-NS-GRPO 2 50
sh eval_pass@k.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 1 0
# sh eval_pass@k.sh chunli-peng/OpenRS-GRPO 3 50
# sh eval_pass@k.sh Qwen/Qwen2.5-1.5B 1 0
# sh eval_pass@k.sh chunli-peng/Qwen2.5-1.5B-NS-GRPO 2 50

echo "Evaluation completed at: $(date)"

