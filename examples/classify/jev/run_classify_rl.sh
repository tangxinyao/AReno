#!/usr/bin/env bash
# Drive one iteration of the classify_rl loop against the in-house sts2_sim
# backend. Intended entry point for running on a DGX Spark workstation (or
# any single-GPU box with the AReno env prepared per the top-level README).
#
# Overridable via env vars:
#   EPISODES         — rollout episode count (default 256)
#   BASE_CKPT        — model to warm-start from (default Qwen/Qwen3-0.6B)
#   MODEL_HUB        — "modelscope" (recommended on Spark) or "hf"
#   OUT_DIR          — root for rollouts + checkpoints (default ./runs/sts2_sim_classify_rl)
#   EPOCHS           — PPO passes over the rollout batch (default 3)
#   MAX_STEPS        — trainer-side max optimizer steps (default 100)
#   DECISIONS_PER_STEP — batch size per optimizer step (default 32)
#   POLICY           — rollout policy, "random" or "greedy" (default random)
#   PYTHON           — python interpreter to use (default python3.11, falls back to python3)
#
# Example:
#   EPISODES=512 BASE_CKPT=inclusionai/ling-3.0-tiny ./run_classify_rl.sh
#
# Produces:
#   ${OUT_DIR}/rollouts/round_<ts>.jsonl   — training-ready decisions
#   ${OUT_DIR}/ckpts/round_<ts>/           — trained score-head checkpoint
#   ${OUT_DIR}/logs/round_<ts>.log         — combined stdout/stderr

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
cd "${ROOT}"

EPISODES="${EPISODES:-256}"
BASE_CKPT="${BASE_CKPT:-Qwen/Qwen3-0.6B}"
MODEL_HUB="${MODEL_HUB:-modelscope}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/sts2_sim_classify_rl}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-100}"
DECISIONS_PER_STEP="${DECISIONS_PER_STEP:-32}"
POLICY="${POLICY:-random}"
PYTHON="${PYTHON:-}"

if [ -z "${PYTHON}" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON="python3.11"
  else
    PYTHON="python3"
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
JSONL="${OUT_DIR}/rollouts/round_${TS}.jsonl"
CKPT="${OUT_DIR}/ckpts/round_${TS}"
LOG="${OUT_DIR}/logs/round_${TS}.log"
mkdir -p "$(dirname "${JSONL}")" "$(dirname "${CKPT}")" "$(dirname "${LOG}")"

echo "=== classify_rl on sts2_sim ===" | tee -a "${LOG}"
echo "Timestamp:          ${TS}"                    | tee -a "${LOG}"
echo "Python:             ${PYTHON}"                | tee -a "${LOG}"
echo "Episodes:           ${EPISODES}"              | tee -a "${LOG}"
echo "Policy:             ${POLICY}"                | tee -a "${LOG}"
echo "Base ckpt:          ${BASE_CKPT}"             | tee -a "${LOG}"
echo "Model hub:          ${MODEL_HUB}"             | tee -a "${LOG}"
echo "Epochs:             ${EPOCHS}"                | tee -a "${LOG}"
echo "Max steps:          ${MAX_STEPS}"             | tee -a "${LOG}"
echo "Decisions/step:     ${DECISIONS_PER_STEP}"    | tee -a "${LOG}"
echo "Rollout jsonl:      ${JSONL}"                 | tee -a "${LOG}"
echo "Save path:          ${CKPT}"                  | tee -a "${LOG}"
echo "Log file:           ${LOG}"                   | tee -a "${LOG}"
echo                                                 | tee -a "${LOG}"

echo "[env check] GPU availability"                 | tee -a "${LOG}"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG}"
import torch
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")
PY
echo                                                 | tee -a "${LOG}"

echo "[1/2] rollout -> ${JSONL}"                    | tee -a "${LOG}"
"${PYTHON}" "${HERE}/rollout_sts2_sim.py" \
  --episodes "${EPISODES}" \
  --out "${JSONL}" \
  --policy "${POLICY}" \
  2>&1 | tee -a "${LOG}"
echo                                                 | tee -a "${LOG}"

echo "[2/2] train -> ${CKPT}"                       | tee -a "${LOG}"
"${PYTHON}" "${HERE}/train_rl.py" \
  --decisions "${JSONL}" \
  --ckpt "${BASE_CKPT}" \
  --model-hub "${MODEL_HUB}" \
  --save-path "${CKPT}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --decisions-per-step "${DECISIONS_PER_STEP}" \
  2>&1 | tee -a "${LOG}"
echo                                                 | tee -a "${LOG}"

echo "Done. Checkpoint saved to ${CKPT}"            | tee -a "${LOG}"
