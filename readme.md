# Spot Reinforcement Learning

Simulate, train, and teleoperate a Boston Dynamics Spot–like quadruped in MuJoCo using PPO from Stable-Baselines3. This repository includes everything needed to:

- build a physics environment with textured ground and Spot meshes via `robot_descriptions`
- train high-performance locomotion policies with vectorized environments and observation/reward normalisation
- replay learned behaviours interactively via a keyboard-driven teleoperation loop
- inspect learning progress through TensorBoard logs

---

## Project Layout

```
spot reinforcement learning/
├── main.py                    # Entry point with --mode train|teleop
├── scripts/
│   ├── train.py               # CLI wrapper for the PPO training pipeline
│   └── teleop.py              # Human-in-the-loop visualisation & control
├── spot_rl/
│   ├── config.py              # Centralised hyper-parameters & paths
│   ├── envs/                  # MuJoCo env, reward, commands, obs builder
│   ├── training/pipeline.py   # VecEnv construction & PPO helpers
│   └── teleop/keyboard.py     # Keyboard command handler (pynput)
├── models/                    # Saved PPO checkpoints (zip)
├── stats/                     # VecNormalize statistics for evaluation
└── spot_tensorboard_advanced/ # TensorBoard event files
```

---

## Installation

```bash
# (Optional) to avoid common errors
sudo apt update
sudo apt update

# (Optional) create a fresh env
python -m venv rl_env
source rl_env/bin/activate  # Windows: rl_env\Scripts\activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:

- `pynput` is only required for teleoperation but bundled in `requirements.txt`.
- GPU training uses whatever device is exposed to PyTorch; otherwise `TRAINING.device` defaults to CPU (see `spot_rl/config.py`).

---

## Configuration

All tunables live in `spot_rl/config.py`:

- `COMMAND`: sampling ranges for linear/angular velocity targets and resampling cadence.
- `SIMULATION`: frame skip, target torso height, max episode length.
- `TRAINING`: PPO hyper-parameters (timesteps, n_steps, batch size, lr, etc.).
- `PATHS`: output locations for the trained model, normalisation stats, and TensorBoard logs. Update these if you want to version checkpoints (`models/ppo_spot_v*.zip`, `stats/vec_normalize_stats_v*.pkl`).

---

## Training a Policy

```bash
# From the project root
python scripts/train.py
# or equivalently
python main.py --mode train
```

What happens:

- `spot_rl.training.pipeline.build_training_env()` creates one `SubprocVecEnv` worker per CPU core and enables observation/reward normalisation.
- `create_model()` builds a Stable-Baselines3 PPO agent with the hyper-parameters defined in `config.py`.
- After `TRAINING.total_timesteps`, both the model weights and VecNormalize statistics are saved to `PATHS.model_path` and `PATHS.stats_path`.

Cancel-safe restarts: training can be interrupted and resumed from the latest ZIP/Pickle pair using SB3’s `PPO.load` if needed.

---

## Monitoring with TensorBoard

Training automatically logs to `spot_tensorboard_advanced/`. Visualise with:

```bash
tensorboard --logdir=./spot_tensorboard_advanced
```

Point your browser to the printed URL (default `http://localhost:6006`) to inspect reward curves, losses, and action statistics.

---

## Teleoperation & Evaluation

Once a policy and normalisation stats exist:

```bash
python scripts/teleop.py
# or: python main.py --mode teleop
```

Workflow:

1. `load_policy_for_teleop()` loads the checkpoint and VecNormalize stats in evaluation mode (`training=False`, `norm_reward=False`).
2. The environment switches to manual command mode so velocity goals are set via the keyboard.
3. `spot_rl.teleop.keyboard.KeyboardController` maps:
   - `w/s`: forward/back linear velocity
   - `a/d`: left/right lateral velocity
   - `q/e`: rotate left/right
   - `8`: stop loop
4. On-screen status displays the live command values. The loop restarts the episode whenever the environment terminates or hits the time limit.

Troubleshooting:

- If you see `pynput` import errors, ensure it is installed and that the script has permission to capture keyboard events (some desktop environments require focus).
- `PATHS.stats_path` must exist; otherwise teleop will raise `FileNotFoundError`. Train once before teleoperating.

---

## Tips & Best Practices

- **Dataset management**: keep multiple versions of `models/*.zip` and `stats/*.pkl` so you can roll back or compare experiments.
- **Render modes**: `spot_rl.envs.spot_env.make_env(render_mode="human")` enables MuJoCo’s viewer; training uses headless mode for performance.
- **Custom rewards/observations**: tweak `spot_rl/envs/reward_calculator.py` or `observation_builder.py` and retrain to experiment with locomotion behaviours.
- **Manual command ranges**: adjust `COMMAND` ranges in the config to explore faster gaits or to enforce safer boundaries during teleop.

---

## License & Attribution

- MuJoCo assets are provided via the `robot_descriptions` project—respect their license.
- Stable-Baselines3 is released under the MIT License.
- This repository is intended for educational and research purposes.
