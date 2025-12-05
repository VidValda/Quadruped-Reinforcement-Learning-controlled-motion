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


def load_policy_for_teleop() -> Tuple[PPO, VecNormalize]:
    if not PATHS.model_path.exists():
        raise FileNotFoundError(
            f"Model file '{PATHS.model_path}' not found. Run scripts/train.py first."
        )

    env = build_visualization_env()
    model = PPO.load(str(PATHS.model_path), env=env, device=TRAINING.device)
    return model, env


