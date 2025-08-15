#!/usr/bin/env bash
set -euo pipefail

# User-configurable settings
RUN_MODE="distributed"   # "single" or "distributed"
GPUS="0,1,2,3"            # e.g., "0" or "0,1,2,4"; for single, first id is used
NUM_PROC=""               # optional override; if empty, derived from number of GPUS

DATA_JSONL="/project/fyp25_hc2/data/eval.jsonl"
MODEL_PATH="/project/fyp25_hc2/results/jina_test_run_fred/finetuned"
BASE_MODEL_PATH="/project/fyp25_hc2/jina-embeddings-v4"
BATCH_SIZE=4
DEVICE="cuda"
SAVE_DIR="/project/fyp25_hc2/results/infer"
SAVE_TOPK=false
TOPK=10
PROMPT_NAME="query"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # expected to be SmartEmbed
ENTRYPOINT="${REPO_ROOT}/jina/infer/cross_retrieval_infer.py"

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
echo "Save dir:    ${SAVE_DIR}"
echo "Save topk:   ${SAVE_TOPK} (k=${TOPK})"
echo "Prompt:      ${PROMPT_NAME}"

COMMON_ARGS=(
  --data_jsonl "${DATA_JSONL}"
  --model_path "${MODEL_PATH}"
  --base_model_path "${BASE_MODEL_PATH}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --save_dir "${SAVE_DIR}"
  --topk "${TOPK}"
  --prompt_name "${PROMPT_NAME}"
)

if [[ "${SAVE_TOPK}" == "true" ]]; then
  COMMON_ARGS+=(--save_topk)
fi

if [[ "${RUN_MODE}" == "single" ]]; then
  if [[ "${DEVICE}" == cuda* && "${#GPU_ARR[@]}" -ge 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
  fi
  python -u "${ENTRYPOINT}" "${COMMON_ARGS[@]}"
else
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  torchrun --standalone --nproc-per-node="${NUM_PROC}" "${ENTRYPOINT}" "${COMMON_ARGS[@]}"
fi


