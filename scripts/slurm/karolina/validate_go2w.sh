#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p qgpu
#SBATCH --gres=gpu:5
#SBATCH --time=12:00:00
#SBATCH --job-name=go2w_validate
#SBATCH --output=logs/slurm/validate/%x_%j.out

# NOTE: all 5 policies (4 depth-student ablations + teacher) run fully in
# parallel, one GPU each, no barrier. Teacher has no depth camera so it
# finishes its 4-scenario x N-seed sequence well before GPU 0-3, but it
# gets its own GPU from t=0 instead of running serially afterward — zero
# idle GPU time. If only 4 GPUs are available, drop back to --gres=gpu:4,
# remove the "teacher 4" line below, and chain teacher onto one of the
# ablation GPUs instead (see meluxina/validate_go2w.sh for that pattern).

# Extra args are passed through to run_validation.py, e.g. 3-seed run:
#   OUT_NAME=validation_$(date +%Y%m%d) sbatch <this script> --seeds 42 43 44 --maze_episodes 100
cd $HOME/go2w

# Re-home SLURM output into a per-job directory (job ID is not known before submission,
# so the directory cannot be pre-created in the SBATCH header).
JOB_LOG_DIR="logs/slurm/validate/${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"
exec > "$JOB_LOG_DIR/${SLURM_JOB_NAME}.out" 2>&1

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
        --num_envs 32 \
        "$@"
}

# All 5 policies in parallel (GPUs 0-4), no barrier.
_run_policy 0 baseline "$@" &
_run_policy 1 longhist "$@" &
_run_policy 2 sparse   "$@" &
_run_policy 3 4cam     "$@" &
_run_policy 4 teacher  "$@" &

wait

# Plot once all 5 policies are done
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

echo "[validate_go2w] Done. Results: logs/nav_play/$OUT_NAME"
