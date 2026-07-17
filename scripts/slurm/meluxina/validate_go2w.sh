#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=go2w_validate
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/validate/%x_%A_%a.out

# Array task → scenario mapping.
# Submit with: sbatch validate_go2w.sh [extra run_validation args]
# Use submit_validation.sh to also chain a plot job automatically.

SCENARIOS=("maze_train" "maze_static" "maze_dynamic" "maze_success")
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"

# Extra args are passed through to run_validation.py, e.g. 3-seed run:
#   OUT_NAME=validation_$(date +%Y%m%d) sbatch <this script> --seeds 42 43 44 --maze_episodes 100
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

# All 4 GPUs start at once with no shared barrier. GPU 0 chains baseline
# then teacher instead of running teacher alone after GPUs 1-3 finish —
# teacher has no depth camera so it completes noticeably faster than a
# depth-student ablation, so stacking it after baseline keeps GPU 0's
# total time close to GPU 1-3's single-ablation time instead of leaving
# 3 GPUs idle during a separate teacher-only tail.
( _run_policy 0 baseline "$@" && _run_policy 0 teacher "$@" ) &
_run_policy 1 longhist "$@" &
_run_policy 2 sparse   "$@" &
_run_policy 3 4cam     "$@" &
wait

echo "[validate] Scenario $SCENARIO done."
