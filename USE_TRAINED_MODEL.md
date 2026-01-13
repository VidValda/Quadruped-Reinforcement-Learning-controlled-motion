# Using Your Trained Model

This guide shows you how to use the trained model `final_model_3.zip` that's in your `models/` directory.

## Quick Start

The model path is already configured in `spot_rl/config.py`. You can use it in two ways:

### Option 1: Test the Model

First, test that the model loads correctly:

```bash
python scripts/test_model.py
```

This will:

- Check if the model file exists
- Try to load it (with or without VecNormalize stats)
- Test a few inference steps
- Show you the model's observation and action spaces

### Option 2: Use with Teleop (Keyboard Control)

Use the model with keyboard control:

```bash
python scripts/teleop.py
```

**Controls:**

- **W/S**: Forward/Backward
- **A/D**: Strafe Left/Right
- **Q/E**: Turn Left/Right
- **I/K**: Increase/Decrease Height
- **J**: Jump
- **R**: Reset Jump
- **8**: Stop

## Model Information

- **Location**: `models/final_model_3.zip`
- **Type**: PPO (Proximal Policy Optimization)
- **Observation Space**: 48 dimensions
- **Action Space**: 12 dimensions (one per joint)

## Loading the Model in Your Own Code

### Basic Loading

```python
from stable_baselines3 import PPO
from spot_rl.config import PATHS, TRAINING

# Load the model
model = PPO.load(str(PATHS.model_path), device=TRAINING.device)

# Use it for inference
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### With MuJoCo Environment

```python
from spot_rl.training.pipeline import load_policy_for_teleop

model, env = load_policy_for_teleop()
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
obs, reward, done, info = env.step(action)
```

### With Genesis Environment

If your model was trained with Genesis:

```python
import genesis as gs
from spot_genesis_env import SpotGenesisEnv, GenesisSB3Wrapper
from stable_baselines3 import PPO

# Initialize Genesis
gs.init(logging_level="warning", backend=gs.constants.backend.cpu)

# Create environment (use same config as training)
env_base = SpotGenesisEnv(
    num_envs=1,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
    show_viewer=True,
    device="cpu"
)
env = GenesisSB3Wrapper(env_base)

# Load model
model = PPO.load("models/final_model_3.zip", env=env, device="cpu")

# Use it
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
obs, reward, done, info = env.step(action)
```

## Troubleshooting

### Model File Not Found

If you get an error that the model file is not found:

1. Check that the file exists:

   ```bash
   ls -lh models/final_model_3.zip
   ```

2. Update the path in `spot_rl/config.py`:
   ```python
   model_path: Path = ROOT_DIR / "models" / "final_model_3.zip"
   ```

### VecNormalize Stats Missing

If your model was trained with Genesis, it might not have VecNormalize stats. The `load_policy_for_teleop()` function will automatically try to load without stats if stats are missing.

### Device Mismatch

If you get CUDA/CPU errors:

1. Check your device in `spot_rl/config.py`:

   ```python
   device: str = "cpu"  # or "cuda"
   ```

2. Or specify when loading:
   ```python
   model = PPO.load(path, device="cpu")
   ```

### Observation/Action Space Mismatch

If you get errors about observation or action space mismatches:

- Make sure the environment configuration matches what was used during training
- Check that `spot_rl/config.py` has the same settings as when the model was trained
- For Genesis models, ensure the observation space is 48 dimensions

## Model Compatibility

Models trained with:

- **MuJoCo**: Work with `scripts/teleop.py` and standard MuJoCo environments
- **Genesis**: Work with Genesis environments (see `test_colab.py` for example)

Both use the same observation/reward structure, so they should be compatible with the same inference code.

## Next Steps

1. **Test the model**: Run `python scripts/test_model.py`
2. **Try teleop**: Run `python scripts/teleop.py` for interactive control
3. **Evaluate performance**: Create your own evaluation script
4. **Fine-tune**: Continue training from this checkpoint if needed
