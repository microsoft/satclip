#!/bin/bash
# =============================================================================
# Local Run Script for M1 Mac / Development
# =============================================================================
#
# Usage:
#   ./run.sh                                      # Default config
#   ./run.sh experiments/elevation.yaml           # Specific config
#   ./run.sh --fast-dev-run                       # Quick test (1 batch)
#   ./run.sh --dummy                              # Dummy run (minimal samples)
#
# =============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================="
echo "Learned Activations - Local Run"
echo -e "==============================================${NC}"

# Parse arguments
CONFIG=""
EXTRA_ARGS=""
DUMMY_MODE=false

for arg in "$@"; do
    case $arg in
        --dummy)
            DUMMY_MODE=true
            ;;
        --fast-dev-run)
            EXTRA_ARGS="${EXTRA_ARGS} --trainer.fast_dev_run=true"
            ;;
        *.yaml)
            CONFIG=$arg
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $arg"
            ;;
    esac
done

# Default config
if [ -z "${CONFIG}" ]; then
    CONFIG="experiments/configs/experiments/checkerboard.yaml"
fi

# Dummy mode: very small experiment for testing
if [ "$DUMMY_MODE" = true ]; then
    echo -e "${YELLOW}Running in DUMMY mode (minimal samples)${NC}"
    EXTRA_ARGS="${EXTRA_ARGS} --data.n_samples=1000 --training.max_epochs=5"
fi

# Change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../../.."

# Detect device
if python -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
    echo -e "${GREEN}Using MPS (Apple Silicon GPU)${NC}"
    EXTRA_ARGS="${EXTRA_ARGS} --hardware.accelerator=mps"
elif python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}Using CUDA GPU${NC}"
    EXTRA_ARGS="${EXTRA_ARGS} --hardware.accelerator=gpu"
else
    echo -e "${YELLOW}Using CPU${NC}"
    EXTRA_ARGS="${EXTRA_ARGS} --hardware.accelerator=cpu"
fi

# Print config
echo "Config: ${CONFIG}"
echo "Extra args: ${EXTRA_ARGS}"
echo ""

# Run training
python -m experiments.train \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

echo -e "${GREEN}=============================================="
echo "Training complete!"
echo -e "==============================================${NC}"
