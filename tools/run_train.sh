#!/usr/bin/env bash
set -euo pipefail

# User-configurable settings
RUN_MODE="distributed"   # "single" or "distributed"
GPUS="0,1,2,3,4,5,6,7"            # e.g., "0" or "0,1,2,3"; for single, first id is used
NUM_PROC=""               # optional override; if empty, derived from number of GPUS

# Optional override paths (usually set in project_config.yaml)
TRAIN_DATA="/project/fyp25_hc2/data/train.jsonl"             # leave empty to use default from tools/train.py
EVAL_DATA=""              # optional
OUTPUT_DIR="/project/fyp25_hc2/jina_test_run_fred"             # Optional: Override for the training output directory. If empty, uses default from config.

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # expected to be SmartEmbed
ENTRYPOINT="${REPO_ROOT}/tools/train.py"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ -z "${NUM_PROC}" || "${NUM_PROC}" == "0" ]]; then
  NUM_PROC="${#GPU_ARR[@]}"
fi

echo "Repo:        ${REPO_ROOT}"
echo "Entry:       ${ENTRYPOINT}"
echo "Run mode:    ${RUN_MODE}"
echo "GPUs:        ${GPUS} (nproc=${NUM_PROC})"

COMMON_ARGS=()
if [[ -n "${TRAIN_DATA}" ]]; then
  COMMON_ARGS+=(--train_data "${TRAIN_DATA}")
fi
if [[ -n "${EVAL_DATA}" ]]; then
  COMMON_ARGS+=(--eval_data "${EVAL_DATA}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  COMMON_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ "${RUN_MODE}" == "single" ]]; then
  if [[ "${#GPU_ARR[@]}" -ge 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
  fi
  python -u "${ENTRYPOINT}" "${COMMON_ARGS[@]}"
else
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  torchrun --standalone --nproc-per-node="${NUM_PROC}" "${ENTRYPOINT}" "${COMMON_ARGS[@]}"
fi


