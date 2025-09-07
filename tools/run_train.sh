#!/usr/bin/env bash
set -euo pipefail

# Load runtime configuration from unified config system
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load default runtime configuration
eval $(python "${SCRIPT_DIR}/get_config.py" --section runtime)

# User-configurable settings (override defaults if needed)
RUN_MODE="${RUN_MODE:-$DEFAULT_RUN_MODE}"   # "single" or "distributed"
GPUS="${GPUS:-$DEFAULT_GPUS}"               # e.g., "0" or "0,1,2,3"
NUM_PROC="${NUM_PROC:-}"                    # optional override; if empty, derived from GPUS

# Load data configuration
eval $(python "${SCRIPT_DIR}/get_config.py" --section data)

# Data paths are now loaded from config as TRAIN_DATA, EVAL_DATA etc.
# They can still be overridden by environment variables if needed

# Load training configuration for output directory  
eval $(python "${SCRIPT_DIR}/get_config.py" --section training)

# OUTPUT_DIR is now loaded from config
# It can still be overridden by environment variables if needed

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
  python -m torch.distributed.run --standalone --nproc-per-node="${NUM_PROC}" "${ENTRYPOINT}" "${COMMON_ARGS[@]}"
fi


