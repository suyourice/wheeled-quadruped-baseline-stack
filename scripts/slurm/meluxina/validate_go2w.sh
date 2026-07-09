#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=go2w_validate
#SBATCH --output=logs/slurm/validate/%x_%j.out

# NOTE: 4 depth student ablations run in parallel (1 GPU each).
# Teacher runs sequentially on GPU 0 after the 4 ablations complete.
# If 5 GPUs are available, change --gres=gpu:5 and uncomment the 5th
# parallel line below to run all 5 policies simultaneously.

cd $HOME/go2w

# Re-home SLURM output into a per-job directory.
JOB_LOG_DIR="logs/slurm/validate/${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"
exec > "$JOB_LOG_DIR/${SLURM_JOB_NAME}.out" 2>&1

module load Apptainer

OUT_NAME="${OUT_NAME:-validation_${SLURM_JOB_ID}}"

_run_policy() {
    local gpu_id=$1
    local policy=$2
    shift 2

    local POLICY_LOG="$JOB_LOG_DIR/${policy}.log"
    exec > "$POLICY_LOG" 2>&1

    local KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_JOB_ID}_gpu${gpu_id}"
    # Pre-create stage/cache dirs to prevent Isaac Sim scene-creation errors.
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
        --env CUDA_VISIBLE_DEVICES=$gpu_id \
        --env GO2W_TERRAIN_CACHE_DIR="$KIT_RUNTIME/terrain_cache" \
        $HOME/isaacsim.sif \
        /isaac-sim/python.sh \
        scripts/run_validation.py \
        --ablation "$policy" \
        --out_name "$OUT_NAME" \
        --skip_plot \
        --num_envs 48 \
        "$@"
}

# Run 4 ablations in parallel (GPUs 0-3)
_run_policy 0 baseline "$@" &
_run_policy 1 longhist "$@" &
_run_policy 2 sparse   "$@" &
_run_policy 3 4cam     "$@" &
# Uncomment and change --gres=gpu:5 above to run teacher in parallel:
# _run_policy 4 teacher "$@" &

wait

# Teacher runs sequentially on GPU 0 (reused after ablations finish)
_run_policy 0 teacher "$@"

# Plot once all 5 policies are done
KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_JOB_ID}_plot"
mkdir -p \
    "$KIT_RUNTIME/cache" \
    "$KIT_RUNTIME/data_documents/Kit/apps/Isaac-Sim/scripts/new_stage"
[[ -s "$KIT_RUNTIME/user.config.json" ]] || printf '%s\n' '{}' > "$KIT_RUNTIME/user.config.json"

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
    "logs/nav_play/$OUT_NAME"

echo "[validate_go2w] Done. Results: logs/nav_play/$OUT_NAME"
