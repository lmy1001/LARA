#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

: "${LARA_VIT_MAE_PATH:?Set LARA_VIT_MAE_PATH to a ViT-MAE-Large checkpoint}"

python scripts/pretrain_moto_tokenizer.py --config-name train_tokenizer "$@"
