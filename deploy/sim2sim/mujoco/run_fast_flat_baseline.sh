#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export GO2W_DISABLE_NON_FOOT_LEG_COLLISIONS="${GO2W_DISABLE_NON_FOOT_LEG_COLLISIONS:-1}"

SIM_TIME="${SIM_TIME:-30.0}"
VX="${VX:-0.5}"
VY="${VY:-0.0}"
WZ="${WZ:-0.0}"
MAX_VX="${MAX_VX:-1.0}"
MAX_VY="${MAX_VY:-1.0}"
MAX_WZ="${MAX_WZ:-0.5}"
WHEEL_SIGN="${WHEEL_SIGN:-1.0}"
VIEWER="${VIEWER:-0}"
REALTIME="${REALTIME:-0}"

EXTRA_ARGS=()
if [[ "${VIEWER}" == "1" ]]; then
  EXTRA_ARGS+=(--viewer)
fi
if [[ "${REALTIME}" == "1" ]]; then
  EXTRA_ARGS+=(--realtime)
fi

python -m deploy.sim2sim.mujoco.policy_rollout_mujoco \
  "${EXTRA_ARGS[@]}" \
  --sim-time "${SIM_TIME}" \
  --vx "${VX}" \
  --vy "${VY}" \
  --wz "${WZ}" \
  --max-vx "${MAX_VX}" \
  --max-vy "${MAX_VY}" \
  --max-wz "${MAX_WZ}" \
  --wheel-sign "${WHEEL_SIGN}" \
  --disable-target-clip
