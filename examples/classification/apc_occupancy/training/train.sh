#!/bin/bash
# Launcher script for APC occupancy detection training.
#
# Usage:
#   bash training/train.sh --config training/configs/zeroshot.yaml
#   bash training/train.sh --config training/configs/finetune-head.yaml --device cpu
#
# Run from the apc_occupancy directory:
#   cd examples/classification/apc_occupancy
#   bash training/train.sh --config training/configs/zeroshot.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default values
CONFIG=""
DEVICE=""
SEED=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: bash training/train.sh --config training/configs/zeroshot.yaml"
    exit 1
fi

# Build command
CMD="python training/train.py --config $CONFIG"
[[ -n "$DEVICE" ]] && CMD="$CMD --device $DEVICE"
[[ -n "$SEED" ]] && CMD="$CMD --seed $SEED"
[[ -n "$EXTRA_ARGS" ]] && CMD="$CMD $EXTRA_ARGS"

echo "============================================"
echo "APC Occupancy Detection - MantisV2"
echo "============================================"
echo "Config: $CONFIG"
echo "Command: $CMD"
echo "============================================"

eval "$CMD"
