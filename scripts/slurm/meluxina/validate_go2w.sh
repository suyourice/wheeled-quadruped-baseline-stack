#!/bin/bash -l
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=go2w_validate
#SBATCH --output=logs/slurm/%x_%j.out

module load Apptainer

cd $HOME/go2w

OUT_NAME="${OUT_NAME:-validation_${SLURM_JOB_ID}}"

_run_ablation() {
    local gpu_id=$1
    local ablation=$2

    local KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_JOB_ID}_gpu${gpu_id}"
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
        $HOME/isaacsim.sif \
        /isaac-sim/python.sh \
        scripts/run_validation.py \
        --ablation "$ablation" \
        --out_name "$OUT_NAME" \
        --skip_plot \
        --num_envs 48 \
        "$@"
}

_run_ablation 0 baseline "$@" &
_run_ablation 1 longhist "$@" &
_run_ablation 2 sparse "$@" &
_run_ablation 3 4cam "$@" &

wait

# Plot once all 4 ablations are done
KIT_RUNTIME="${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/${SLURM_JOB_ID}_plot"
mkdir -p "$KIT_RUNTIME/cache" "$KIT_RUNTIME/data_documents/Kit/apps/Isaac-Sim/scripts/new_stage"
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
