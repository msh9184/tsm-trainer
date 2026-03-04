#!/usr/bin/env bash
# Thin shell orchestrator — sets up the environment and delegates to Python.
#
# Usage:
#   ./download_and_prepare.sh --root_dir /group-volume/ts-dataset --datasets all
#   ./download_and_prepare.sh --root_dir /group-volume/ts-dataset --datasets chronos_valid --subset chronos-lite --stage 2
#
#   # GiftEvalPretrain — full pipeline (download → convert → validate)
#   ./download_and_prepare.sh \
#       --root_dir /group-volume/ts-dataset \
#       --datasets gift-eval-pretrain
#
#   # GiftEvalPretrain — convert + validate only (if snapshot already downloaded)
#   ./download_and_prepare.sh \
#       --root_dir /group-volume/ts-dataset \
#       --datasets gift-eval-pretrain \
#       --stage 2 \
#       --num_proc 4 \
#       --resume
#
# Output paths for GiftEvalPretrain:
#   Snapshot (raw)   : {root_dir}/gift_eval_pretrain_raw/
#   Arrow (training) : {root_dir}/chronos_datasets/gift_eval_pretrain/
#
# Environment variables (set before calling this script):
#   PYTHON          Python interpreter to use (default: python3)
#   HF_TOKEN        HuggingFace authentication token
#   HTTPS_PROXY     HTTPS proxy URL (if behind a proxy)
#   HTTP_PROXY      HTTP proxy URL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON="${PYTHON:-python3}"

# Add repo root to PYTHONPATH so that relative imports work
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON" "$SCRIPT_DIR/download_and_prepare.py" "$@"
