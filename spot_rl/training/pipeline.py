from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Tuple

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from spot_rl.config import PATHS, TRAINING
from spot_rl.envs.spot_env import make_env


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_training_env() -> VecNormalize:
    num_cpu = multiprocessing.cpu_count()
    env_fns = [lambda: make_env(render_mode=None) for _ in range(num_cpu)]
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


def create_model(env: VecNormalize) -> PPO:
    policy_kwargs = TRAINING.policy_kwargs.copy() if TRAINING.policy_kwargs else {}
    if 'activation_fn' in policy_kwargs and isinstance(policy_kwargs['activation_fn'], str):
        activation_name = policy_kwargs.pop('activation_fn')
        if activation_name == 'elu':
            policy_kwargs['activation_fn'] = nn.ELU
        else:
            policy_kwargs['activation_fn'] = nn.ReLU
    
    batch_size = TRAINING.batch_size if TRAINING.batch_size > 0 else TRAINING.n_steps // 4
    
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(PATHS.tensorboard_log),
        n_steps=TRAINING.n_steps,
        batch_size=batch_size,
        n_epochs=TRAINING.n_epochs,
        learning_rate=TRAINING.learning_rate,
        gamma=TRAINING.gamma,
        gae_lambda=TRAINING.gae_lambda,
        clip_range=TRAINING.clip_range,
        ent_coef=TRAINING.ent_coef,
        vf_coef=TRAINING.vf_coef,
        max_grad_norm=TRAINING.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=TRAINING.device,
    )


def train():
    env = build_training_env()
    model = create_model(env)

    model.learn(total_timesteps=TRAINING.total_timesteps)

    _ensure_parent(PATHS.model_path)
    _ensure_parent(PATHS.stats_path)
    model.save(str(PATHS.model_path))
    env.save(str(PATHS.stats_path))
    env.close()


def load_policy_for_teleop(require_stats: bool = False) -> Tuple[PPO, VecNormalize]:
    """
    Load a trained policy for teleoperation.
    Automatically detects if model is Genesis-trained and uses appropriate environment.
    
    Args:
        require_stats: If True, requires VecNormalize stats file. If False, 
                     will try to load without stats (for Genesis-trained models).
    """
    if not PATHS.model_path.exists():
        raise FileNotFoundError(
            f"Model file '{PATHS.model_path}' not found. "
            f"Expected at: {PATHS.model_path}"
        )

    # First, try to inspect the model to see if it's Genesis-trained
    try:
        temp_model = PPO.load(str(PATHS.model_path), device=TRAINING.device)
        is_genesis = (
            hasattr(temp_model.action_space, 'low') and 
            hasattr(temp_model.action_space, 'high') and
            len(temp_model.action_space.low) > 0 and
            temp_model.action_space.low[0] == -1.0 and 
            temp_model.action_space.high[0] == 1.0
        )
        del temp_model  # Free memory
        
        if is_genesis:
            print("Detected Genesis-trained model. Using Genesis environment...")
            return _load_genesis_model()
    except Exception:
        pass  # If inspection fails, try normal loading
    
    # Try to load with stats first (for MuJoCo-trained models)
    if PATHS.stats_path.exists() or require_stats:
        try:
            env = build_visualization_env()
            model = PPO.load(str(PATHS.model_path), env=env, device=TRAINING.device)
            return model, env
        except Exception as e:
            if require_stats:
                raise e
            print(f"Warning: Could not load with VecNormalize stats: {e}")
            print("Attempting to load without stats...")
    
    # Load without VecNormalize (for Genesis-trained models)
    try:
        env_base = DummyVecEnv([lambda: make_env(render_mode="human")])
        model = PPO.load(str(PATHS.model_path), env=env_base, device=TRAINING.device)
        return model, env_base
    except Exception as e:
        # Last resort: try Genesis
        print(f"Warning: Could not load with MuJoCo: {e}")
        print("Attempting to load with Genesis environment...")
        return _load_genesis_model()


def _load_genesis_model() -> Tuple[PPO, any]:
    """Load model with Genesis environment."""
    try:
        import genesis as gs
        from spot_genesis_env import SpotGenesisEnv, GenesisSB3Wrapper
        
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        scripts_path = str(PROJECT_ROOT / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        from train_genesis import get_genesis_configs
        
        # Initialize Genesis
        gs.init(logging_level="warning", backend=gs.constants.backend.cpu)
        
        # Get configs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_genesis_configs()
        
        # Create Genesis environment
        # Enable viewer for teleop so user can see the robot
        env_base = SpotGenesisEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,  # Enable viewer for teleop visualization
            device="cpu"
        )
        env = GenesisSB3Wrapper(env_base)
        
        # Load model
        model = PPO.load(str(PATHS.model_path), env=env, device=TRAINING.device)
        return model, env
    except ImportError as e:
        raise ImportError(
            f"Genesis environment required but not available. "
            f"Install with: pip install genesis-world\n"
            f"Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model with Genesis environment: {e}\n"
            f"Make sure the model was trained with Genesis and the environment config matches."
        )


