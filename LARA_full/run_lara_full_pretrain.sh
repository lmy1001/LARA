#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

: "${LARA_BASE_MODEL_PATH:?Set LARA_BASE_MODEL_PATH to the GR00T N1.5 checkpoint}"
: "${LARA_TOKENIZER_PATH:?Set LARA_TOKENIZER_PATH to the pretrained motion tokenizer}"
: "${LARA_VIT_MAE_PATH:?Set LARA_VIT_MAE_PATH to the tokenizer ViT-MAE checkpoint}"

exec python scripts/gr00t_finetune.py --config-name mani_test_all "$@"
