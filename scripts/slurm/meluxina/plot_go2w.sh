#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=go2w_plot
#SBATCH --output=logs/slurm/validate/%x_%j.out

# Runs after validate_go2w.sh array completes (chained via --dependency in submit_validation.sh).

cd $HOME/go2w

module load Apptainer

OUT_NAME="${OUT_NAME:?OUT_NAME not set}"

KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_JOB_ID}_plot"
mkdir -p \
    "$KIT_RUNTIME/cache" \
    "$KIT_RUNTIME/data_documents/Kit/apps/Isaac-Sim/scripts/new_stage"
[[ -s "$KIT_RUNTIME/user.config.json" ]] || printf '%s\n' '{}' > "$KIT_RUNTIME/user.config.json"

# maze_success (single-route success/SPL) and the long-horizon scenarios
# (sustained multi-route progress) measure different things and must not
# share a chart's axes or a summary.csv's rows — plot_validation.py enforces
# this via --out_prefix. Plot each group separately, same as run_validation.py's
# own internal auto-plot logic.
apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind "$KIT_RUNTIME/cache:/isaac-sim/kit/cache" \
    --bind "$KIT_RUNTIME/data_documents:/isaac-sim/kit/data/documents" \
    --bind "$KIT_RUNTIME/user.config.json:/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/user.config.json" \
    --env GIT_PYTHON_REFRESH=quiet \
    --env PYTHONUNBUFFERED=1 \
    $HOME/isaacsim.sif \
    /isaac-sim/python.sh \
    scripts/plot_validation.py \
    "logs/nav_play/$OUT_NAME" \
    --scenarios maze_train maze_static maze_dynamic

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind "$KIT_RUNTIME/cache:/isaac-sim/kit/cache" \
    --bind "$KIT_RUNTIME/data_documents:/isaac-sim/kit/data/documents" \
    --bind "$KIT_RUNTIME/user.config.json:/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/user.config.json" \
    --env GIT_PYTHON_REFRESH=quiet \
    --env PYTHONUNBUFFERED=1 \
    $HOME/isaacsim.sif \
    /isaac-sim/python.sh \
    scripts/plot_validation.py \
    "logs/nav_play/$OUT_NAME" \
    --scenarios maze_success --out_prefix success_

echo "[plot] Done. Results: logs/nav_play/$OUT_NAME"
