#!/usr/bin/env bash
set -euo pipefail

# User-configurable settings
RUN_MODE="distributed"   # "single" or "distributed"
GPUS="0,1,2,3,4,6,7"            # e.g., "0" or "0,1,2,4"; for single, first id is used
NUM_PROC=""               # optional override; if empty, derived from number of GPUS

DATA_JSONL="/project/fyp25_hc2/data/eval.jsonl"
MODEL_PATH="/project/fyp25_hc2/results/jina_test_run_fred/finetuned"
BASE_MODEL_PATH="/project/fyp25_hc2/jina-embeddings-v4"
BATCH_SIZE=4
DEVICE="cuda"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # expected to be SmartEmbed
ENTRYPOINT="${REPO_ROOT}/jina/eval/cross_retrieval_eval.py"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ -z "${NUM_PROC}" || "${NUM_PROC}" == "0" ]]; then
  NUM_PROC="${#GPU_ARR[@]}"
fi

echo "Repo:        ${REPO_ROOT}"
echo "Entry:       ${ENTRYPOINT}"
echo "Run mode:    ${RUN_MODE}"
echo "GPUs:        ${GPUS} (nproc=${NUM_PROC})"
echo "Data JSONL:  ${DATA_JSONL}"
echo "Model path:  ${MODEL_PATH}"
echo "Base model:  ${BASE_MODEL_PATH}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Device:      ${DEVICE}"

if [[ "${RUN_MODE}" == "single" ]]; then
  # Use the first GPU id if device is cuda
  if [[ "${DEVICE}" == cuda* && "${#GPU_ARR[@]}" -ge 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
  fi
  python -u "${ENTRYPOINT}" \
    --data_jsonl "${DATA_JSONL}" \
    --model_path "${MODEL_PATH}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}"
else
  # Distributed via torchrun over the specified GPU list
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  torchrun --standalone --nproc-per-node="${NUM_PROC}" "${ENTRYPOINT}" \
    --data_jsonl "${DATA_JSONL}" \
    --model_path "${MODEL_PATH}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}"
fi


