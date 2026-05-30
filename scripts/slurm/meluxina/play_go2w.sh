#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=go2w_play
#SBATCH --output=logs/slurm/%x_%j.out

module load Apptainer

cd $HOME/go2w

apptainer exec --nv \
  --bind $HOME:$HOME \
  --env GIT_PYTHON_REFRESH=quiet \
  $HOME/isaacsim.sif \
  /isaac-sim/python.sh \
  scripts/rsl_rl/play.py \
  "$@"
