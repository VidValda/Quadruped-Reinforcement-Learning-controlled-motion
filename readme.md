# Spot Reinforcement Learning

Simulate, train, and teleoperate a Boston Dynamics Spot–like quadruped in MuJoCo using PPO from Stable-Baselines3. This repository includes everything needed to:

- Build a physics environment with textured ground and Spot meshes via `robot_descriptions`
- Train high-performance locomotion policies with vectorized environments and observation/reward normalization
- Replay learned behaviors interactively via a keyboard-driven teleoperation loop
- Inspect learning progress through TensorBoard logs with detailed reward component tracking

---

## Project Layout

```
spot reinforcement learning/
├── main.py                    # Entry point with --mode train|teleop
├── scripts/
│   ├── train.py               # CLI wrapper for the PPO training pipeline
│   ├── teleop.py              # Human-in-the-loop visualization & control
│   └── sensor_monitor.py      # Sensor monitoring utilities
├── spot_rl/
│   ├── config.py              # Centralized hyper-parameters & paths
│   ├── envs/
│   │   ├── spot_env.py        # Main Gymnasium environment (CustomSpotEnv)
│   │   ├── model_loader.py    # MuJoCo model loading with floor/texture setup
│   │   ├── command_manager.py # Velocity command sampling and manual control
│   │   ├── observation_builder.py # Observation space construction
│   │   ├── reward_calculator.py   # Multi-component reward function
│   │   ├── info_wrapper.py    # Info dict collection wrapper
│   │   └── utils.py           # Utility functions (quaternion, velocity transforms)
│   ├── training/
│   │   ├── pipeline.py        # VecEnv construction, PPO model creation, training loop
│   │   └── callbacks.py       # TensorBoard metrics callback
│   └── teleop/
│       └── keyboard.py        # Keyboard command handler (pynput)
├── models/                    # Saved PPO checkpoints (zip files)
├── stats/                     # VecNormalize statistics for evaluation (pkl files)
└── spot_tensorboard_advanced/ # TensorBoard event files and monitor logs
```

---

## Installation

```bash
# (Optional) create a fresh virtual environment
python -m venv rl_env
source rl_env/bin/activate  # Windows: rl_env\Scripts\activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Dependencies:**

- `gymnasium`: Gym API for RL environments
- `mujoco`: MuJoCo physics simulator
- `stable-baselines3[extra]`: PPO implementation and vectorized environments
- `torch`: PyTorch for neural network training
- `robot_descriptions`: Spot robot model and meshes
- `pynput`: Keyboard input for teleoperation (optional, only needed for teleop mode)

**Notes:**

- `pynput` is only required for teleoperation but bundled in `requirements.txt`.
- GPU training uses whatever device is exposed to PyTorch; otherwise `TRAINING.device` defaults to CPU (see `spot_rl/config.py`).
- The project uses `SubprocVecEnv` for parallel training, which scales with CPU core count.

---

## Configuration

All tunables live in `spot_rl/config.py`:

### `COMMAND` (CommandConfig)

- `lin_vel_x_range`: Linear velocity range in x-direction (default: `(-0.5, 1.0)`)
- `lin_vel_y_range`: Linear velocity range in y-direction (default: `(-0.3, 0.3)`)
- `ang_vel_range`: Angular velocity range (default: `(-0.3, 0.3)`)
- `resampling_time_s`: Time interval for resampling velocity commands during training (default: `6.0`)

### `SIMULATION` (SimulationConfig)

- `frame_skip`: Number of simulation steps per action (default: `5`)
- `target_height`: Target torso height in meters (default: `0.5247`)
- `max_episode_steps`: Maximum steps per episode (default: `4000`)
- `initial_position`: Starting position `(x, y, z)` (default: `(0.0, 0.0, 0.55)`)

### `TRAINING` (TrainingConfig)

- `total_timesteps`: Total training timesteps (default: `15_000_000`)
- `n_steps`: Steps per rollout (default: `1024`)
- `batch_size`: Batch size for PPO updates (default: `4096`)
- `n_epochs`: Number of optimization epochs per update (default: `10`)
- `learning_rate`: Learning rate (default: `1e-4`)
- `gamma`: Discount factor (default: `0.99`)
- `gae_lambda`: GAE lambda parameter (default: `0.95`)
- `clip_range`: PPO clip range (default: `0.2`)
- `ent_coef`: Entropy coefficient (default: `0.0`)
- `device`: Training device `"cpu"` or `"cuda"` (default: `"cpu"`)
- `num_envs`: Number of parallel environments (default: `None`, uses CPU count)

### `PATHS` (PathConfig)

- `model_path`: Path to save/load trained model (default: `models/ppo_spot_v44.zip`)
- `stats_path`: Path to save/load VecNormalize statistics (default: `stats/vec_normalize_stats_v44.pkl`)
- `tensorboard_log`: TensorBoard log directory (default: `spot_tensorboard_advanced`)

**Versioning:** Update version numbers in `PATHS` to create new checkpoints (`ppo_spot_v*.zip`, `vec_normalize_stats_v*.pkl`).

---

## Training a Policy

```bash
# From the project root
python scripts/train.py
# or equivalently
python main.py --mode train
```

**What happens:**

1. **Environment Setup**: `build_training_env()` creates `SubprocVecEnv` with one worker per CPU core (or `TRAINING.num_envs` if specified). Each environment is wrapped with:

   - `InfoCollectorWrapper`: Captures info dicts for custom metrics
   - `Monitor`: Logs episode statistics for TensorBoard
   - `VecNormalize`: Normalizes observations and rewards

2. **Model Creation**: `create_model()` builds a Stable-Baselines3 PPO agent with:

   - MLP policy network
   - Hyper-parameters from `TRAINING` config
   - TensorBoard logging enabled

3. **Training Loop**: The model trains for `TRAINING.total_timesteps` with:

   - Custom `TensorBoardMetricsCallback` logging reward components
   - Automatic episode statistics collection via Monitor wrapper

4. **Saving**: After training completes:
   - Model weights saved to `PATHS.model_path`
   - VecNormalize statistics saved to `PATHS.stats_path`
   - TensorBoard logs written to `PATHS.tensorboard_log`

**Cancel-safe restarts:** Training can be interrupted and resumed. To resume, modify `config.py` to point to an existing checkpoint and use SB3's `PPO.load()`.

---

## Monitoring with TensorBoard

Training automatically logs to `spot_tensorboard_advanced/`. Visualize with:

```bash
tensorboard --logdir=./spot_tensorboard_advanced
```

Point your browser to the printed URL (default `http://localhost:6006`) to inspect:

- **Reward Components**: Individual reward terms (linear velocity, angular velocity, orientation, control costs, foot clearance, etc.)
- **Tracking Metrics**: Velocity tracking errors
- **Performance Metrics**: Action rates, joint velocities
- **Gait Metrics**: Stance/swing phase tracking, foot clearance
- **Standard PPO Metrics**: Policy loss, value loss, entropy, explained variance
- **Episode Statistics**: Mean episode reward, length, success rate

The callback in `spot_rl/training/callbacks.py` extracts detailed metrics from the reward calculator's info dict.

---

## Teleoperation & Evaluation

Once a policy and normalization stats exist:

```bash
python scripts/teleop.py
# or: python main.py --mode teleop
```

**Workflow:**

1. **Loading**: `load_policy_for_teleop()` loads:

   - Model checkpoint from `PATHS.model_path`
   - VecNormalize statistics from `PATHS.stats_path`
   - Environment in evaluation mode (`training=False`, `norm_reward=False`)

2. **Manual Control**: The environment switches to manual command mode via `enable_manual_control()`.

3. **Keyboard Controls**: `KeyboardController` maps:

   - `w/s`: Increase/decrease forward linear velocity (`lin_x`)
   - `a/d`: Increase/decrease lateral linear velocity (`lin_y`)
   - `q/e`: Increase/decrease angular velocity (`ang_z`)
   - `8`: Stop the teleoperation loop
   - `ESC`: Exit (alternative stop method)

4. **Visualization**: MuJoCo's passive viewer displays the robot in real-time.

5. **Episode Management**: The loop automatically resets the episode when:
   - Environment terminates (e.g., torso height too low)
   - Episode time limit reached

**Status Display**: On-screen status shows current command values, updated in real-time.

**Troubleshooting:**

- If you see `pynput` import errors, ensure it is installed: `pip install pynput`
- Some desktop environments require the terminal window to have focus for keyboard capture
- `PATHS.stats_path` must exist; otherwise teleop will raise `FileNotFoundError`. Train once before teleoperating.
- Ensure the model file exists at `PATHS.model_path` before running teleop.

---

## Environment Details

### Observation Space

The observation includes:

- Joint positions (excluding root)
- Joint velocities (excluding root)
- Local root velocities (linear and angular in robot frame)
- Torso height
- Pitch and roll angles
- Target linear velocity (2D)
- Target angular velocity (scalar)

### Action Space

- Box space with shape `(num_actuators,)` and range `[-0.5, 0.5]`
- Actions are added to the default homing pose to produce final joint positions
- Final actions are clipped to `[-2π, 2π]`

### Reward Function

The reward calculator (`spot_rl/envs/reward_calculator.py`) includes:

- **Velocity Tracking**: Exponential rewards for matching target linear/angular velocities
- **Height Penalty**: Quadratic penalty for deviation from target torso height
- **Orientation Penalty**: Penalty for roll/pitch deviations
- **Control Costs**: Penalties for action magnitude and action rate
- **Joint Velocity Penalty**: Penalty for excessive joint velocities
- **Nominal Pose Penalty**: Penalty for deviation from default homing pose
- **Foot Clearance Reward**: Bell-curve reward for lifting feet during swing phase (encourages natural gait)

**Termination**: Episode terminates if torso height drops below `0.26m` (configurable via `termination_height_threshold`).

### Command Management

During training, velocity commands are randomly sampled from `COMMAND` ranges and resampled every `resampling_time_s` seconds. There's a 20% chance of sampling zero velocity (stop command). During teleoperation, commands are set manually via keyboard input.

---

## Tips & Best Practices

- **Version Management**: Keep multiple versions of `models/*.zip` and `stats/*.pkl` so you can roll back or compare experiments. Update version numbers in `config.py`.

- **Render Modes**:

  - Training uses `render_mode=None` (headless) for performance
  - Teleoperation uses `render_mode="human"` for visualization
  - You can test with `render_mode="human"` in training, but it will be slower

- **Custom Rewards/Observations**:

  - Tweak `spot_rl/envs/reward_calculator.py` to experiment with locomotion behaviors
  - Modify `spot_rl/envs/observation_builder.py` to change the observation space
  - Adjust reward weights in `SpotRewardCalculator.__init__()` to emphasize different behaviors

- **Command Ranges**: Adjust `COMMAND` ranges in the config to:

  - Explore faster gaits (increase velocity ranges)
  - Enforce safer boundaries during teleop (reduce ranges)
  - Train for specific behaviors (e.g., forward-only locomotion)

- **Parallel Training**: The number of parallel environments defaults to CPU count. Set `TRAINING.num_envs` explicitly if you want fewer environments (useful for debugging or resource-constrained systems).

- **GCP Deployment**: The training pipeline includes helper messages for downloading models from GCP instances (see `pipeline.py`).

---

## License & Attribution

- MuJoCo assets are provided via the `robot_descriptions` project—respect their license.
- Stable-Baselines3 is released under the MIT License.
- This repository is intended for educational and research purposes.
