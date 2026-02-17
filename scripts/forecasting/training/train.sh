#!/bin/bash
# ===========================================================================
# Chronos-2 Multi-Node Training Launcher (mpirun-based)
# ===========================================================================
#
# Auto-detects single-node vs multi-node from /horovod/generated/hostfile.
# Compatible with the Samsung GPU server MPI environment.
#
# Usage:
#   # Pretraining (from scratch) - auto-detects nodes from hostfile
#   bash scripts/forecasting/training/train.sh \
#       --config scripts/forecasting/training/configs/chronos2-base.yaml
#
#   # Quick test with limited data
#   bash scripts/forecasting/training/train.sh \
#       --config scripts/forecasting/training/configs/chronos2-test.yaml \
#       --max-steps 50 --max-train-series 1000
#
#   # Resume training
#   bash scripts/forecasting/training/train.sh \
#       --config scripts/forecasting/training/configs/chronos2-base.yaml \
#       --resume-from-checkpoint ./output/chronos2-base-stage1/checkpoint-50000
#
# Environment:
#   HOSTFILE:     Path to hostfile (default: /horovod/generated/hostfile)
#   MASTER_PORT:  Port for distributed rendezvous (default: 29500)
#   NCCL_DEBUG:   NCCL debug level (default: WARN, set to INFO for debugging)
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_REPO="$(cd "${SCRIPT_DIR}" && while [ ! -d "src" ] && [ "$PWD" != "/" ]; do cd ..; done && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_chronos2.py"
HOSTFILE="${HOSTFILE:-/horovod/generated/hostfile}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ---------------------------------------------------------------------------
# Auto-detect distributed configuration
# ---------------------------------------------------------------------------
if [ -f "${HOSTFILE}" ]; then
    NUM_NODES=$(wc -l < "${HOSTFILE}")
    GPUS_PER_NODE=$(head -1 "${HOSTFILE}" | grep -oP 'slots=\K[0-9]+')
    TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

    # Extract MASTER_ADDR from first hostname in hostfile
    FIRST_HOST=$(head -1 "${HOSTFILE}" | awk '{print $1}')
    export MASTER_ADDR="${MASTER_ADDR:-$(python3 -c "import socket; print(socket.gethostbyname('${FIRST_HOST}'))" 2>/dev/null || echo "${FIRST_HOST}")}"
    export MASTER_PORT

    echo "============================================================"
    echo "Chronos-2 Multi-Node Training"
    echo "============================================================"
    echo "  Hostfile:       ${HOSTFILE}"
    echo "  Num nodes:      ${NUM_NODES}"
    echo "  GPUs per node:  ${GPUS_PER_NODE}"
    echo "  Total GPUs:     ${TOTAL_GPUS}"
    echo "  Master addr:    ${MASTER_ADDR}"
    echo "  Master port:    ${MASTER_PORT}"
    echo "  NCCL debug:     ${NCCL_DEBUG}"
    echo "  Train script:   ${TRAIN_SCRIPT}"
    echo "  Arguments:      $@"
    echo "============================================================"

    # MPI-based multi-node execution
    mpirun --allow-run-as-root \
        -np "${TOTAL_GPUS}" \
        -bind-to none \
        -map-by slot \
        --hostfile "${HOSTFILE}" \
        -mca pml ob1 \
        -mca btl ^openib \
        -mca orte_keep_fqdn_hostnames t \
        -x MASTER_ADDR \
        -x MASTER_PORT \
        -x HOSTFILE="${HOSTFILE}" \
        -x NCCL_DEBUG="${NCCL_DEBUG}" \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_DISABLE=1 \
        -x PATH \
        -x PYTHONPATH="${ROOT_REPO}/src:${PYTHONPATH:-}" \
        -x LD_LIBRARY_PATH \
        -x no_proxy \
        -x http_proxy \
        -x https_proxy \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        python3 "${TRAIN_SCRIPT}" "$@"

else
    # Single-node fallback: detect GPUs locally
    GPUS_PER_NODE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

    echo "============================================================"
    echo "Chronos-2 Single-Node Training"
    echo "============================================================"
    echo "  No hostfile found at ${HOSTFILE}"
    echo "  GPUs detected:  ${GPUS_PER_NODE}"
    echo "  Train script:   ${TRAIN_SCRIPT}"
    echo "  Arguments:      $@"
    echo "============================================================"

    if [ "${GPUS_PER_NODE}" -gt 1 ]; then
        # Multi-GPU single node via torchrun
        export PYTHONPATH="${ROOT_REPO}/src:${PYTHONPATH:-}"
        torchrun --nproc_per_node="${GPUS_PER_NODE}" \
            "${TRAIN_SCRIPT}" "$@"
    else
        # Single GPU
        export PYTHONPATH="${ROOT_REPO}/src:${PYTHONPATH:-}"
        python3 "${TRAIN_SCRIPT}" "$@"
    fi
fi
