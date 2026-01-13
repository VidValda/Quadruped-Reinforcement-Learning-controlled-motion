# Genesis Training Guide

This guide explains how to use the Genesis GPU-accelerated simulation for training Spot, which can be used outside of Google Colab.

## Overview

The Genesis training script (`scripts/train_genesis.py`) provides GPU-accelerated parallel simulation that can train much faster than the standard MuJoCo-based training. It uses the same observation and reward structure as the main project, ensuring compatibility.

## Installation

1. **Install Genesis World**:

   ```bash
   pip install genesis-world
   ```

   Or install all requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. **GPU Requirements** (for GPU training):

   - CUDA-capable GPU (NVIDIA)
   - CUDA toolkit installed
   - PyTorch with CUDA support

   For CPU-only training, you can use `--backend cpu` and `--device cpu`.

## Usage

### Basic Training

```bash
python scripts/train_genesis.py
```

### Custom Configuration

```bash
python scripts/train_genesis.py \
    --num-envs 4096 \
    --max-steps 150000000 \
    --exp-name my_experiment \
    --device cuda:0 \
    --backend gpu
```

### CPU Training (if no GPU available)

```bash
python scripts/train_genesis.py \
    --num-envs 512 \
    --device cpu \
    --backend cpu
```

### Arguments

- `--num-envs`: Number of parallel environments (default: 4096)

  - Adjust based on GPU VRAM: 1024 for T4, 4096+ for A100
  - CPU: Use 512-1024 for reasonable performance

- `--max-steps`: Total training timesteps (default: 150M)

- `--exp-name`: Experiment name for TensorBoard logs (default: spot_genesis_sb3)

- `--device`: Device for Genesis simulation (default: cuda:0)

  - Use `cuda:0`, `cuda:1`, etc. for GPU
  - Use `cpu` for CPU-only

- `--backend`: Genesis backend (default: gpu)

  - `gpu`: GPU-accelerated (requires CUDA)
  - `cpu`: CPU-only (slower but works everywhere)

- `--show-viewer`: Enable 3D viewer (significantly slows training)

- `--batch-size`: PPO batch size (default: auto-calculated)

## Configuration

The script automatically uses configuration from `spot_rl/config.py`:

- `COMMAND`: Command sampling ranges
- `SIMULATION`: Simulation parameters (KP, KD, episode length, etc.)
- `REWARD`: Reward scales and parameters
- `OBS`: Observation scales
- `TRAINING`: PPO hyperparameters

## Output

- **Model checkpoints**: Saved in `spot_tensorboard_advanced/{exp_name}/spot_ppo_*.zip`
- **Final model**: `spot_tensorboard_advanced/{exp_name}/final_model.zip`
- **TensorBoard logs**: `spot_tensorboard_advanced/{exp_name}/`
- **Project model copy**: `models/ppo_spot_genesis.zip` (for compatibility with teleop)

## Loading Trained Models

Models trained with Genesis are compatible with the standard teleop script:

```bash
# Update spot_rl/config.py to point to your model:
# model_path: Path = ROOT_DIR / "models" / "ppo_spot_genesis.zip"

python scripts/teleop.py
```

Or load directly in Python:

```python
from stable_baselines3 import PPO
model = PPO.load("spot_tensorboard_advanced/spot_genesis_sb3/final_model.zip")
```

## Differences from Colab Code

The Colab code (`testgenesisrl.py`) can be used outside Colab, but this project's script provides:

1. **Integration with project config**: Uses `spot_rl/config.py` instead of hardcoded values
2. **Command-line interface**: Proper argparse instead of Colab's class-based args
3. **Project structure**: Follows the project's file organization
4. **Compatibility**: Models work with existing teleop and evaluation scripts

## Performance Tips

1. **GPU Memory**: If you get OOM errors, reduce `--num-envs`
2. **CPU Training**: Much slower; use fewer environments (512-1024)
3. **Viewer**: Only enable `--show-viewer` for debugging; it significantly slows training
4. **Batch Size**: Auto-calculated from `n_steps`, but you can override with `--batch-size`

## Troubleshooting

### Genesis Import Error

```bash
pip install genesis-world
```

### CUDA Out of Memory

Reduce `--num-envs` or use CPU backend:

```bash
--num-envs 1024 --device cpu --backend cpu
```

### Joint Name Errors

The script uses joint names from `robot_descriptions`. If you see joint name errors, check that `robot_descriptions` is installed and the Spot model is available.

### Model Compatibility

Models trained with Genesis use the same observation/reward structure as MuJoCo training, so they're fully compatible with the teleop script and evaluation tools.
