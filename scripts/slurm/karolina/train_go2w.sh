#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p qgpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --job-name=go2w_train
#SBATCH --output=logs/slurm/%x_%j.out

cd $HOME/go2w

KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_JOB_ID:-local_train}"
mkdir -p \
  "$KIT_RUNTIME/cache" \
  "$KIT_RUNTIME/data_documents/Kit/apps/Isaac-Sim/scripts/new_stage" \
  "$HOME/go2w/Documents/Kit/shared"
if [[ ! -s "$KIT_RUNTIME/user.config.json" ]]; then
  printf '%s\n' '{}' > "$KIT_RUNTIME/user.config.json"
fi

apptainer exec --nv \
  --bind $HOME:$HOME \
  --bind "$KIT_RUNTIME/cache:/isaac-sim/kit/cache" \
  --bind "$KIT_RUNTIME/data_documents:/isaac-sim/kit/data/documents" \
  --bind "$KIT_RUNTIME/user.config.json:/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/user.config.json" \
  --env GIT_PYTHON_REFRESH=quiet \
  --env PYTHONUNBUFFERED=1 \
  $HOME/isaacsim.sif \
  /isaac-sim/python.sh \
  scripts/rsl_rl/train.py \
  "$@"
