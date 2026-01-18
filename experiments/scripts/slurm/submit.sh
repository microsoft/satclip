#!/bin/bash
#SBATCH -J learned_activations        # Job name
#SBATCH -o logs/%x_%j.out             # Output file
#SBATCH -e logs/%x_%j.err             # Error file
#SBATCH -p condo-jacobsn              # Partition (your lab's partition)
#SBATCH --gpus a40:1                  # Request 1 A40 GPU
#SBATCH -A engr-lab-jacobsn           # Account
#SBATCH -t 4:00:00                    # Time limit (4 hours)
#SBATCH --mem=32G                     # Memory
#SBATCH -n 1                          # Number of tasks
#SBATCH -c 8                          # CPUs per task

# =============================================================================
# SLURM Job Script for Learned Activations Experiments
# =============================================================================
#
# Usage:
#   sbatch submit.sh                              # Run with default config
#   sbatch submit.sh experiments/elevation.yaml   # Run with specific config
#   sbatch submit.sh --encoding sh_l40 --activation spline  # Override options
#
# To request different resources:
#   sbatch -p condo-jacobsn,general-gpu --gpus 1 submit.sh  # Any GPU
#   sbatch --time=8:00:00 submit.sh                          # Longer time
#
# =============================================================================

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Start time: $(date)"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Option 1: Conda environment
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate satclip
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate satclip
fi

# Option 2: Virtual environment
# source /path/to/venv/bin/activate

# Option 3: Module system
# module load cuda/11.8
# module load python/3.10

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

# =============================================================================
# Run Training
# =============================================================================

# Change to project directory
cd "${SLURM_SUBMIT_DIR}/../.."

# Default config
CONFIG="${1:-experiments/configs/experiments/elevation.yaml}"

# Run training
python -m experiments.train \
    --config "${CONFIG}" \
    "${@:2}"

# Print completion
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
