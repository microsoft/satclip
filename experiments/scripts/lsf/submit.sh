#!/bin/bash
# =============================================================================
# LSF Job Script for Learned Activations Experiments
# =============================================================================
#
# Usage:
#   bsub < submit.sh                    # Submit with default config
#   bsub -J myexp < submit.sh          # Custom job name
#
# For interactive jobs:
#   bsub -gpu "num=1" -m osprey -Is /bin/bash
#
# =============================================================================

#BSUB -J learned_activations           # Job name
#BSUB -o logs/%J.out                   # Output file
#BSUB -e logs/%J.err                   # Error file
#BSUB -q gpu-compute                   # Queue
#BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA40"
#BSUB -n 8                             # Number of cores
#BSUB -R "rusage[mem=4000]"            # Memory per core (4GB x 8 = 32GB)
#BSUB -W 4:00                          # Wall time (4 hours)

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "LSF Job ID: ${LSB_JOBID}"
echo "Host: ${LSB_HOSTS}"
echo "Start time: $(date)"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Conda activation
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate satclip
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate satclip
fi

# Print environment
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# =============================================================================
# Run Training
# =============================================================================

cd "${LS_SUBCWD}/../.."

# Default config
CONFIG="${1:-experiments/configs/experiments/elevation.yaml}"

python -m experiments.train \
    --config "${CONFIG}"

echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
