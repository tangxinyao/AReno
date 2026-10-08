#!/usr/bin/env bash
# Iterative classify_rl PPO loop against the in-house sts2_sim backend.
#
# For round 0 the behavior policy is uniform-random (cold start). Starting
# from round 1 the previous round's checkpoint is served through
# serve_decisions.py and the rollout script samples decisions from that
# policy at `TEMPERATURE`, writing the correct `old_logp` for off-policy
# correction in the subsequent PPO step. train_rl.py warm-starts from
# ckpt_{round-1} instead of BASE_CKPT once the loop has produced a ckpt.
#
# Overridable via env vars:
#   ROUNDS              — total loop iterations (default 3)
#   EPISODES            — rollout episodes per round (default 256)
#   BASE_CKPT           — model to warm-start round 0 from (default Qwen/Qwen3-0.6B)
#   MODEL_HUB           — "modelscope" or "hf" (default modelscope)
#   OUT_DIR             — root for rollouts + checkpoints (default ./runs/sts2_sim_classify_rl)
#   EPOCHS              — PPO passes over the round's rollout batch (default 3)
#   MAX_STEPS           — trainer max optimizer steps per round (default 100)
#   DECISIONS_PER_STEP  — batch size per optimizer step (default 32)
#   TEMPERATURE         — sampling temperature for policy server (default 1.0)
#   BASE_PORT           — first serve_decisions port; round k uses BASE_PORT+k (default 8125)
#   HEALTH_RETRIES      — /health poll retries at 2s each (default 90 = 3 min)
#   PYTHON              — python interpreter (default python3.11, falls back to python3)
#
# Produces per round:
#   ${OUT_DIR}/rollouts/round_<i>.jsonl
#   ${OUT_DIR}/ckpts/round_<i>/        <- train_rl.py --save-path target
#   ${OUT_DIR}/logs/round_<i>.log
#   ${OUT_DIR}/logs/server_<i>.log     (round >= 1)
#
# Example on DGX Spark:
#   ROUNDS=5 EPISODES=512 BASE_CKPT=inclusionai/ling-3.0-tiny \
#     ./examples/classify/jev/run_classify_rl.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
cd "${ROOT}"

ROUNDS="${ROUNDS:-3}"
EPISODES="${EPISODES:-256}"
BASE_CKPT="${BASE_CKPT:-Qwen/Qwen3-0.6B}"
MODEL_HUB="${MODEL_HUB:-modelscope}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/sts2_sim_classify_rl}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-100}"
DECISIONS_PER_STEP="${DECISIONS_PER_STEP:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BASE_PORT="${BASE_PORT:-8125}"
HEALTH_RETRIES="${HEALTH_RETRIES:-90}"
PYTHON="${PYTHON:-}"

if [ -z "${PYTHON}" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON="python3.11"
  else
    PYTHON="python3"
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}/rollouts" "${OUT_DIR}/ckpts" "${OUT_DIR}/logs"

RUN_LOG="${OUT_DIR}/logs/run_${TS}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=== classify_rl iterative loop on sts2_sim ==="
echo "Timestamp:         ${TS}"
echo "Python:            ${PYTHON}"
echo "Rounds:            ${ROUNDS}"
echo "Episodes/round:    ${EPISODES}"
echo "Policy temp:       ${TEMPERATURE}"
echo "Base ckpt:         ${BASE_CKPT}"
echo "Model hub:         ${MODEL_HUB}"
echo "Epochs:            ${EPOCHS}"
echo "Max steps:         ${MAX_STEPS}"
echo "Decisions/step:    ${DECISIONS_PER_STEP}"
echo "Out dir:           ${OUT_DIR}"
echo

echo "[env check] GPU availability"
"${PYTHON}" - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")
PY
echo

# ---------------------------------------------------------------------------
# Helpers

SERVER_PID=""
cleanup_server() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping serve_decisions pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

wait_for_health() {
  local url="$1"
  local retries="$2"
  for i in $(seq 1 "${retries}"); do
    if curl -sf --max-time 2 "${url}/health" >/dev/null 2>&1; then
      echo "[health] ready after ${i} attempt(s)"
      return 0
    fi
    sleep 2
  done
  echo "[health] FAILED after ${retries} attempts (${url})"
  return 1
}

latest_step_dir() {
  local base="$1"
  # Prefer step_XXXXXX subdirs; fall back to the save root if the trainer
  # only wrote a flat checkpoint.
  local latest
  latest="$(find "${base}" -maxdepth 1 -type d -name 'step_*' -print 2>/dev/null | sort -V | tail -1)"
  if [ -n "${latest}" ]; then
    echo "${latest}"
  else
    echo "${base}"
  fi
}

# ---------------------------------------------------------------------------
# Loop

PREV_CKPT=""
for ROUND in $(seq 0 $((ROUNDS - 1))); do
  echo
  echo "============================================================"
  echo "  ROUND ${ROUND} / $((ROUNDS - 1))"
  echo "============================================================"

  JSONL="${OUT_DIR}/rollouts/round_${ROUND}.jsonl"
  CKPT_OUT="${OUT_DIR}/ckpts/round_${ROUND}"
  ROUND_LOG="${OUT_DIR}/logs/round_${ROUND}.log"

  if [ -z "${PREV_CKPT}" ]; then
    echo "[round ${ROUND}] cold start — uniform random rollout"
    POLICY_ARGS=(--policy random)
    TRAIN_FROM="${BASE_CKPT}"
  else
    PORT=$((BASE_PORT + ROUND))
    SERVER_URL="http://127.0.0.1:${PORT}"
    SERVER_LOG="${OUT_DIR}/logs/server_${ROUND}.log"
    echo "[round ${ROUND}] starting serve_decisions with ckpt=${PREV_CKPT} on port ${PORT}"
    nohup "${PYTHON}" "${HERE}/serve_decisions.py" \
      --checkpoint "${PREV_CKPT}" \
      --port "${PORT}" \
      --temperature "${TEMPERATURE}" \
      > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "[round ${ROUND}] server pid=${SERVER_PID}, log=${SERVER_LOG}"

    if ! wait_for_health "${SERVER_URL}" "${HEALTH_RETRIES}"; then
      echo "[round ${ROUND}] server never became healthy; last lines of its log:"
      tail -n 50 "${SERVER_LOG}" || true
      cleanup_server
      exit 1
    fi

    POLICY_ARGS=(
      --policy server
      --server-url "${SERVER_URL}"
      --temperature "${TEMPERATURE}"
    )
    TRAIN_FROM="${PREV_CKPT}"
  fi

  echo "[round ${ROUND}] rollout -> ${JSONL}"
  "${PYTHON}" "${HERE}/rollout_sts2_sim.py" \
    "${POLICY_ARGS[@]}" \
    --episodes "${EPISODES}" \
    --out "${JSONL}" \
    --seed-start "$((ROUND * EPISODES))" \
    2>&1 | tee -a "${ROUND_LOG}"

  cleanup_server

  echo "[round ${ROUND}] train warm-start-from=${TRAIN_FROM} -> ${CKPT_OUT}"
  "${PYTHON}" "${HERE}/train_rl.py" \
    --decisions "${JSONL}" \
    --ckpt "${TRAIN_FROM}" \
    --model-hub "${MODEL_HUB}" \
    --save-path "${CKPT_OUT}" \
    --epochs "${EPOCHS}" \
    --max-steps "${MAX_STEPS}" \
    --decisions-per-step "${DECISIONS_PER_STEP}" \
    2>&1 | tee -a "${ROUND_LOG}"

  PREV_CKPT="$(latest_step_dir "${CKPT_OUT}")"
  echo "[round ${ROUND}] done. next-round policy ckpt: ${PREV_CKPT}"
done

echo
echo "=== loop complete ==="
echo "Final checkpoint: ${PREV_CKPT}"
echo "Full run log:     ${RUN_LOG}"
