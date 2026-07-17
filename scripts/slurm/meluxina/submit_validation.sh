#!/bin/bash
# Submit hospital maze ablation validation on Meluxina.
#
# Usage:
#   bash scripts/slurm/meluxina/submit_validation.sh [OUT_NAME] [extra sbatch args]
#
# Example:
#   bash scripts/slurm/meluxina/submit_validation.sh v1_model1100
#   bash scripts/slurm/meluxina/submit_validation.sh v1_model1100 --maze_episodes 100
#
# What it does:
#   1. Submits a 4-task array job (maze_train / maze_static / maze_dynamic /
#      maze_success).  The first three retain the long progress protocol;
#      maze_success is the additional short-route success/SPL protocol.
#      Each task uses 4 GPUs: 4 ablations in parallel, then teacher sequentially.
#      All 4 tasks run simultaneously → 16 GPUs total, 4× faster than sequential.
#   2. Submits a plot job with --dependency=afterok on the array job.
#      Runs plot_validation.py once all 4 scenarios complete.
#
# Outputs: logs/nav_play/<OUT_NAME>/

set -euo pipefail

OUT_NAME="${1:-validation_$(date +%Y%m%d_%H%M%S)}"
shift || true   # remaining args passed through to run_validation.py via validate_go2w.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."   # repo root

mkdir -p logs/slurm/validate

echo "[submit] OUT_NAME = $OUT_NAME"
echo "[submit] Extra run_validation args: $*"

# 1. Submit the 4-task array job.
ARRAY_JOB_ID=$(sbatch --parsable \
    --export=ALL,OUT_NAME="$OUT_NAME" \
    "$SCRIPT_DIR/validate_go2w.sh" \
    "$@")
echo "[submit] Array job submitted: $ARRAY_JOB_ID (tasks 0-3: train/static/dynamic/success)"

# 2. Submit the plot job, runs only after all 4 array tasks succeed.
PLOT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --export=ALL,OUT_NAME="$OUT_NAME" \
    "$SCRIPT_DIR/plot_go2w.sh")
echo "[submit] Plot job submitted: $PLOT_JOB_ID (runs after array $ARRAY_JOB_ID completes)"

echo ""
echo "[submit] Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/validate/${ARRAY_JOB_ID}/maze_train.log"
echo ""
echo "[submit] Results will appear in: logs/nav_play/$OUT_NAME/"
