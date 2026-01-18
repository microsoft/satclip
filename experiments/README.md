# Learned Activations Experiments

Modular, HPC-ready experiment infrastructure for comparing activation functions and positional encodings for geographic location encoding.

## Quick Start

```bash
# 1. Create environment
conda env create -f experiments/environment.yaml
conda activate satclip

# 2. Test setup (recommended before HPC jobs)
python experiments/scripts/local/test_setup.py

# 3. Run locally (M1 Mac / local GPU)
./experiments/scripts/local/run.sh --dummy  # Quick test
./experiments/scripts/local/run.sh          # Full run

# 4. Submit HPC job (SLURM)
sbatch experiments/scripts/slurm/submit.sh

# 5. Submit sweep (SLURM array job)
sbatch experiments/scripts/slurm/sweep.sh
```

## Project Structure

```
experiments/
├── models/                          # Model components
│   ├── activations/                 # Activation functions
│   │   ├── relu.py                  # ReLU (baseline)
│   │   ├── siren.py                 # SIREN (SatCLIP default)
│   │   ├── spline.py                # Learnable spline
│   │   └── rff.py                   # Random Fourier Features
│   ├── encodings/                   # Positional encodings
│   │   ├── spherical_harmonics.py   # SH (L=10, 20, 40)
│   │   ├── raw.py                   # Normalized lon/lat
│   │   ├── rff_encoding.py          # RFF encoding
│   │   └── cartesian.py             # 3D Cartesian
│   ├── location_encoder.py          # Main encoder
│   └── lightning_module.py          # PyTorch Lightning modules
├── data/                            # Data loading
│   ├── huggingface.py               # SatCLIP from HuggingFace
│   ├── geographic.py                # Elevation, population
│   └── synthetic.py                 # Checkerboard, etc.
├── configs/                         # YAML configurations
│   ├── base.yaml                    # Base config
│   ├── activations/                 # Activation configs
│   ├── encodings/                   # Encoding configs
│   └── experiments/                 # Full experiment configs
├── scripts/                         # Job scripts
│   ├── slurm/                       # SLURM job scripts
│   ├── lsf/                         # LSF job scripts
│   └── local/                       # Local run scripts
├── utils/                           # Utilities
│   └── config.py                    # Config loading/merging
└── train.py                         # Main training script
```

## Configuration System

Configs are composable YAML files. You can combine:
- Base config (defaults)
- Experiment config (task, data)
- Encoding config (SH, raw, RFF)
- Activation config (ReLU, SIREN, spline)

Example:
```bash
python -m experiments.train \
    --config experiments/configs/experiments/elevation.yaml \
    --config experiments/configs/encodings/sh_l10.yaml \
    --config experiments/configs/activations/spline.yaml
```

CLI overrides work too:
```bash
python -m experiments.train \
    --config experiments/configs/experiments/elevation.yaml \
    --model.hidden_dim=512 \
    --training.learning_rate=0.0001 \
    --training.max_epochs=200
```

## Available Experiments

### 1. Elevation Prediction
```bash
python -m experiments.train --config experiments/configs/experiments/elevation.yaml
```

### 2. Population Density
```bash
python -m experiments.train --config experiments/configs/experiments/population.yaml \
    --data.data_path=/path/to/gpw_file.tif
```

### 3. Checkerboard (Synthetic)
```bash
python -m experiments.train --config experiments/configs/experiments/checkerboard.yaml
```

### 4. Continental/Regional Analysis
```bash
python -m experiments.train --config experiments/configs/experiments/continent.yaml \
    --data.region="[-125, -65, 25, 50]"  # USA
```

### 5. Multi-Resolution Analysis
```bash
python -m experiments.train --config experiments/configs/experiments/resolution.yaml
```

### 6. Contrastive Learning (SatCLIP-style)
```bash
python -m experiments.train --config experiments/configs/experiments/contrastive.yaml
```

## Key Findings from Hamza Notebooks

From extensive experiments (NB14-21):

1. **SH + ReLU beats SIREN** by +2.93%
2. **RFF + SH catastrophically fails** (-7.74%) - frequency interference
3. **Spline beats SIREN** by +1.88% (optimal: k=15, ReLU init)
4. **ReLU still beats Spline** by +0.63%
5. **Zero init for Spline is catastrophic**

Recommendation: Use **SH + ReLU** as baseline, test **Spline** for edge cases.

## HPC Usage

### SLURM (Osprey)
```bash
# Single job
sbatch experiments/scripts/slurm/submit.sh

# Array job (sweep all combinations)
sbatch experiments/scripts/slurm/sweep.sh

# Custom resources
sbatch -p condo-jacobsn,general-gpu --gpus 1 experiments/scripts/slurm/submit.sh
```

### LSF
```bash
bsub < experiments/scripts/lsf/submit.sh
```

### Interactive
```bash
# SLURM
srun -p condo-jacobsn --gpus a40:1 --pty bash

# LSF
bsub -gpu "num=1" -m osprey -Is /bin/bash
```

## Local Development (M1 Mac)

```bash
# Quick test (minimal samples, 5 epochs)
./experiments/scripts/local/run.sh --dummy

# Full checkerboard experiment
./experiments/scripts/local/run.sh

# Specific config
./experiments/scripts/local/run.sh experiments/configs/experiments/elevation.yaml

# Fast dev run (1 batch only)
./experiments/scripts/local/run.sh --fast-dev-run
```

## Extending

### Add New Activation
1. Create `experiments/models/activations/myactivation.py`
2. Add to `__init__.py` registry
3. Create config `experiments/configs/activations/myactivation.yaml`

### Add New Encoding
1. Create `experiments/models/encodings/myencoding.py`
2. Add to `__init__.py` registry
3. Create config `experiments/configs/encodings/myencoding.yaml`

### Add New Dataset
1. Add to `experiments/data/`
2. Update `get_datamodule()` in `train.py`
3. Create experiment config

## Logging

Results are logged to:
- TensorBoard: `logs/experiments/<experiment_name>_<timestamp>/`
- Checkpoints: Same directory

View with:
```bash
tensorboard --logdir logs/
```
