#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi
