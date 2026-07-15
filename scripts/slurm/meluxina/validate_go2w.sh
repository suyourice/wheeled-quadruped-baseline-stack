#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=go2w_validate
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/validate/%x_%A_%a.out

# Array task → scenario mapping.
# Submit with: sbatch validate_go2w.sh [extra run_validation args]
# Use submit_validation.sh to also chain a plot job automatically.

SCENARIOS=("maze_train" "maze_static" "maze_dynamic")
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"

cd $HOME/go2w

JOB_LOG_DIR="logs/slurm/validate/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"
exec > "$JOB_LOG_DIR/${SCENARIO}.log" 2>&1

module load Apptainer

OUT_NAME="${OUT_NAME:-validation_${SLURM_ARRAY_JOB_ID}}"

_run_policy() {
    local gpu_id=$1
    local policy=$2
    shift 2

    local KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_ARRAY_JOB_ID}_${SCENARIO}_gpu${gpu_id}"
    mkdir -p \
        "$KIT_RUNTIME/cache" \
        "$KIT_RUNTIME/data_documents/Kit/apps/Isaac-Sim/scripts/new_stage" \
        "$HOME/go2w/Documents/Kit/shared"
    [[ -s "$KIT_RUNTIME/user.config.json" ]] || printf '%s\n' '{}' > "$KIT_RUNTIME/user.config.json"

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
        --scenario "$SCENARIO" \
        --out_name "$OUT_NAME" \
        --skip_plot \
        --num_envs 48 \
        "$@"
}

echo "[validate] Scenario: $SCENARIO  Job: $SLURM_ARRAY_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"

# 4 depth-student ablations in parallel (GPUs 0-3).
_run_policy 0 baseline "$@" &
_run_policy 1 longhist "$@" &
_run_policy 2 sparse   "$@" &
_run_policy 3 4cam     "$@" &
wait

# Teacher runs sequentially on GPU 0 after ablations finish.
# No depth camera → significantly faster; GPU 0 reuse avoids extra node.
_run_policy 0 teacher "$@"

echo "[validate] Scenario $SCENARIO done."
