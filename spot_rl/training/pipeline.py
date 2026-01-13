from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from spot_rl.config import PATHS, TRAINING
from spot_rl.envs.spot_env import make_env


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_training_env(
    num_envs: Optional[int] = None,
    tensorboard_log: Optional[str] = None,
) -> VecNormalize:
    """Build vectorized training environment with optional Monitor wrapper for TensorBoard.
    
    Args:
        num_envs: Number of parallel environments. If None, uses CPU count.
        tensorboard_log: Directory for TensorBoard logs. If provided, wraps envs with Monitor.
    
    Returns:
        VecNormalize environment ready for training.
    """
    if num_envs is None:
        num_envs = multiprocessing.cpu_count()
    
    print(f"Creating {num_envs} parallel environments...")
    
    def make_env_fn(rank: int):
        """Create a single environment wrapped with Monitor for TensorBoard logging."""
        env = make_env(render_mode=None)
        # Monitor wrapper is essential for logging rollout metrics like ep_rew_mean
        # It collects episode rewards and lengths for TensorBoard
        if tensorboard_log:
            # Create a unique log directory for each environment
            # Monitor needs a directory to write episode stats that PPO will read
            log_dir = os.path.join(tensorboard_log, f"monitor_env_{rank}")
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir, allow_early_resets=True)
        # If no tensorboard_log, don't wrap with Monitor
        # PPO will still work but won't log rollout metrics to TensorBoard
        return env
    
    # Create environment factory functions for each rank
    # Use default argument to properly capture rank value for multiprocessing
    env_fns = []
    for i in range(num_envs):
        def _make_env(rank=i):
            return make_env_fn(rank)
        env_fns.append(_make_env)
    
    env = SubprocVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=TRAINING.gamma)
    return env


def build_visualization_env(render_mode: str = "human") -> VecNormalize:
    env_base = DummyVecEnv([lambda: make_env(render_mode=render_mode)])
    if not PATHS.stats_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at {PATHS.stats_path}. "
            "Train a model first via scripts/train.py."
        )

    env = VecNormalize.load(PATHS.stats_path, env_base)
    env.training = False
    env.norm_reward = False
    return env


def create_model(
    env: VecNormalize,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """Create PPO model with specified configuration.
    
    Args:
        env: Vectorized normalized environment.
        tensorboard_log: Directory for TensorBoard logs. If None, uses PATHS.tensorboard_log.
    
    Returns:
        Configured PPO model ready for training.
    """
    log_dir = tensorboard_log if tensorboard_log is not None else str(PATHS.tensorboard_log)
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=TRAINING.n_steps,
        batch_size=TRAINING.batch_size,
        n_epochs=TRAINING.n_epochs,
        learning_rate=TRAINING.learning_rate,
        gamma=TRAINING.gamma,
        device=TRAINING.device,
    )


def train():
    """Main training function using configuration from spot_rl.config."""
    # Use values from config
    num_envs = TRAINING.num_envs
    tensorboard_log = str(PATHS.tensorboard_log)
    total_timesteps = TRAINING.total_timesteps
    
    # Ensure tensorboard log directory exists
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Build environment (with Monitor wrapper for TensorBoard metrics)
    env = build_training_env(num_envs=num_envs, tensorboard_log=tensorboard_log)
    
    # Create model
    model = create_model(env, tensorboard_log=tensorboard_log)

    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    _ensure_parent(PATHS.model_path)
    _ensure_parent(PATHS.stats_path)
    model.save(str(PATHS.model_path))
    env.save(str(PATHS.stats_path))
    env.close()
    
    print(f"\nTraining complete!")
    print(f"Model saved: {PATHS.model_path}")
    print(f"Stats saved: {PATHS.stats_path}")


def load_policy_for_teleop() -> Tuple[PPO, VecNormalize]:
    if not PATHS.model_path.exists():
        raise FileNotFoundError(
            f"Model file '{PATHS.model_path}' not found. Run scripts/train.py first."
        )

    env = build_visualization_env()
    model = PPO.load(str(PATHS.model_path), env=env, device=TRAINING.device)
    return model, env


