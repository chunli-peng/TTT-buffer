# ThetaEvolve on SLURM (A100) - Quick Commands

## 1) Build environment once
cd /scratch/user/$USER/TTT-buffer
bash build_conda.sh

## 2) Configure run parameters before submit
# Edit run.sh and set at least:
# - SAVE_PATH
# - SMALL_MODEL_NAME / TASK / CONFIG_POSTFIX
# - WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT

## 3) Submit training job
cd /scratch/user/$USER/TTT-buffer
mkdir -p logs
sbatch hpc_bash/submit_singlenode_train_ns_grpo.sh

## Optional: override resource requests when submitting
# sbatch --time=12:00:00 --gres=gpu:a100:4 --cpus-per-task=32 --mem=256G hpc_bash/submit_singlenode_train_ns_grpo.sh

## 4) Queue / job management
squeue -u $USER
scontrol show job <job_id>
scancel <job_id>

## 5) Follow logs
tail -f $(ls -t logs/thetaevolve_train_*.out | head -n 1)
tail -f $(ls -t logs/thetaevolve_train_*.err | head -n 1)

## 6) Visualize results (per README.md)
cd /scratch/user/$USER/TTT-buffer/Results/CirclePacking
python vis.py

## 7) Useful diagnostics
nvidia-smi
sinfo -p gpu -o "%20N %10c %10m %20G %8D %10P %10t"
sprio -u $USER
