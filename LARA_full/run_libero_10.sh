#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

: "${LARA_STAGE2_CHECKPOINT:?Set LARA_STAGE2_CHECKPOINT to a trained LARA Stage 2 checkpoint}"
: "${LARA_VIT_MAE_PATH:?Set LARA_VIT_MAE_PATH to the tokenizer ViT-MAE checkpoint}"

if [[ -z "${LARA_POSTTRAIN_TOKENIZER_PATH:-}" ]]; then
    LARA_POSTTRAIN_TOKENIZER_PATH="${LARA_STAGE2_CHECKPOINT%/}/latent_motion_tokenizer"
fi
export LARA_POSTTRAIN_TOKENIZER_PATH

exec python scripts/gr00t_finetune.py --config-name lara_posttrain_libero_10 "$@"
