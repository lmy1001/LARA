#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

required_paths=(
    OPENPI_BASE_MODEL_PATH
    OPENPI_ASSETS_DIR
    LARA_TOKENIZER_PATH
    LARA_VIT_MAE_PATH
)

for variable_name in "${required_paths[@]}"; do
    value="${!variable_name:-}"
    if [[ -z "${value}" ]]; then
        echo "Missing required environment variable: ${variable_name}" >&2
        exit 2
    fi
    if [[ ! -e "${value}" ]]; then
        echo "${variable_name} does not exist: ${value}" >&2
        exit 2
    fi
done

if [[ ! -f "${OPENPI_BASE_MODEL_PATH}/model.safetensors" ]]; then
    echo "Missing π0.5 PyTorch weights: ${OPENPI_BASE_MODEL_PATH}/model.safetensors" >&2
    exit 2
fi
if [[ ! -f "${OPENPI_ASSETS_DIR}/libero/norm_stats.json" ]]; then
    echo "Missing LIBERO normalization stats: ${OPENPI_ASSETS_DIR}/libero/norm_stats.json" >&2
    exit 2
fi
if [[ ! -f "${LARA_TOKENIZER_PATH}/config.yaml" && ! -f "${LARA_TOKENIZER_PATH}/config.json" ]]; then
    echo "Tokenizer checkpoint needs config.yaml or config.json: ${LARA_TOKENIZER_PATH}" >&2
    exit 2
fi
if [[ ! -f "${LARA_TOKENIZER_PATH}/model.safetensors" && ! -f "${LARA_TOKENIZER_PATH}/pytorch_model.bin" ]]; then
    echo "Tokenizer checkpoint needs model.safetensors or pytorch_model.bin: ${LARA_TOKENIZER_PATH}" >&2
    exit 2
fi

export LARA_LIBERO_DATASET="${LARA_LIBERO_DATASET:-physical-intelligence/libero}"
export LARA_OUTPUT_DIR="${LARA_OUTPUT_DIR:-${REPO_ROOT}/outputs}"

num_gpus="${LARA_NUM_GPUS:-4}"
exp_name="${LARA_EXP_NAME:-lara_pi05_libero}"
resume="${LARA_RESUME:-0}"

train_args=(
    pi05_lara_libero
    --exp_name "${exp_name}"
)

if [[ "${resume}" == "1" || "${resume}" == "true" ]]; then
    train_args+=(--resume)
fi

exec uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${num_gpus}" \
    scripts/train_pytorch.py \
    "${train_args[@]}" \
    "$@"
