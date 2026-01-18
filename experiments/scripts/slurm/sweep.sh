#!/bin/bash
#SBATCH -J activation_sweep           # Job name
#SBATCH -o logs/sweep_%A_%a.out       # Output file (array job)
#SBATCH -e logs/sweep_%A_%a.err       # Error file
#SBATCH -p condo-jacobsn              # Partition
#SBATCH --gpus a40:1                  # GPU
#SBATCH -A engr-lab-jacobsn           # Account
#SBATCH -t 2:00:00                    # Time per job
#SBATCH --mem=32G                     # Memory
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --array=0-11                  # 12 jobs (3 encodings x 4 activations)

# =============================================================================
# SLURM Array Job for Activation/Encoding Sweep
# =============================================================================
#
# Runs all combinations of:
#   - Encodings: sh_l10, sh_l40, raw
#   - Activations: relu, siren, spline, rff
#
# Usage:
#   sbatch sweep.sh
#   sbatch --array=0-5 sweep.sh  # Run subset
#
# =============================================================================

set -e
mkdir -p logs

# Configuration grid
ENCODINGS=("sh_l10" "sh_l40" "raw")
ACTIVATIONS=("relu" "siren" "spline" "rff")

# Calculate indices
N_ENCODINGS=${#ENCODINGS[@]}
N_ACTIVATIONS=${#ACTIVATIONS[@]}

ENCODING_IDX=$((SLURM_ARRAY_TASK_ID / N_ACTIVATIONS))
ACTIVATION_IDX=$((SLURM_ARRAY_TASK_ID % N_ACTIVATIONS))

ENCODING=${ENCODINGS[$ENCODING_IDX]}
ACTIVATION=${ACTIVATIONS[$ACTIVATION_IDX]}

echo "=============================================="
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Encoding: ${ENCODING}"
echo "Activation: ${ACTIVATION}"
echo "=============================================="

# Skip invalid combinations (RFF activation with SH encoding)
if [[ "${ENCODING}" == "sh_l"* ]] && [[ "${ACTIVATION}" == "rff" ]]; then
    echo "SKIPPING: RFF activation conflicts with SH encoding"
    exit 0
fi

# Environment setup
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate satclip
fi

cd "${SLURM_SUBMIT_DIR}/../.."

# Run training
python -m experiments.train \
    --config experiments/configs/experiments/elevation.yaml \
    --config experiments/configs/encodings/${ENCODING}.yaml \
    --config experiments/configs/activations/${ACTIVATION}.yaml \
    --experiment.name "${ENCODING}_${ACTIVATION}" \
    --experiment.seed ${SLURM_ARRAY_TASK_ID}

echo "Completed: ${ENCODING} + ${ACTIVATION}"
