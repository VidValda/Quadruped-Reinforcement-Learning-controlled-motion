"""
Training script for Spot using Genesis GPU-accelerated simulation.
This script can be used outside of Colab to train models with Genesis.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import genesis as gs
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from spot_genesis_env import SpotGenesisEnv, GenesisSB3Wrapper
from spot_rl.config import COMMAND, SIMULATION, REWARD, OBS, TRAINING, PATHS

# Try to get robot path from robot_descriptions
try:
    from robot_descriptions import spot_mj_description
    SPOT_XML_PATH = spot_mj_description.MJCF_PATH
except ImportError:
    print("Warning: robot_descriptions not found. Falling back to default URDF.")
    SPOT_XML_PATH = "urdf/spot/spot.urdf"


def get_genesis_configs():
    """
    Convert project configs to Genesis environment format.
    """
    # Default joint angles mapping (Spot joint names)
    default_joint_angles = {
        "fl_hx": 0.0, "fl_hy": 0.7, "fl_kn": -1.4,
        "fr_hx": 0.0, "fr_hy": 0.7, "fr_kn": -1.4,
        "hl_hx": 0.0, "hl_hy": 0.7, "hl_kn": -1.4,
        "hr_hx": 0.0, "hr_hy": 0.7, "hr_kn": -1.4,
    }

    env_cfg = {
        "num_actions": 12,
        "robot_path": SPOT_XML_PATH,
        "dof_names": [
            "fl_hx", "fl_hy", "fl_kn",
            "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn",
            "hr_hx", "hr_hy", "hr_kn",
        ],
        "default_joint_angles": default_joint_angles,
        "kp": SIMULATION.kp,
        "kd": SIMULATION.kd,
        "termination_if_roll_greater_than": SIMULATION.termination_if_roll_greater_than,
        "termination_if_pitch_greater_than": SIMULATION.termination_if_pitch_greater_than,
        "base_init_pos": [0.0, 0.0, SIMULATION.target_height],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": SIMULATION.episode_length_s,
        "resampling_time_s": COMMAND.resampling_time_s,
        "action_scale": SIMULATION.action_scale,
        "simulate_action_latency": SIMULATION.simulate_action_latency,
        "clip_actions": SIMULATION.clip_actions,
    }

    obs_cfg = {
        "num_obs": OBS.num_obs,
        "obs_scales": OBS.obs_scales,
    }

    reward_cfg = {
        "tracking_sigma": REWARD.tracking_sigma,
        "base_height_target": REWARD.base_height_target,
        "jump_reward_steps": REWARD.jump_reward_steps,
        "reward_scales": REWARD.reward_scales,
    }

    command_cfg = {
        "num_commands": COMMAND.num_commands,
        "lin_vel_x_range": list(COMMAND.lin_vel_x_range),
        "lin_vel_y_range": list(COMMAND.lin_vel_y_range),
        "ang_vel_range": list(COMMAND.ang_vel_range),
        "height_range": list(COMMAND.height_range),
        "jump_range": list(COMMAND.jump_range),
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser(description="Train Spot using Genesis GPU simulation")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of parallel environments (default: 4096, adjust based on GPU VRAM)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150_000_000,
        help="Total training timesteps (default: 150M)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="spot_genesis_sb3",
        help="Experiment name for logging (default: spot_genesis_sb3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for Genesis simulation (default: cuda:0, use 'cpu' if no GPU)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Genesis backend (default: gpu)",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Enable 3D viewer (slows down training significantly)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for PPO (default: auto-calculated from n_steps)",
    )

    args = parser.parse_args()

    # Initialize Genesis
    backend = gs.constants.backend.gpu if args.backend == "gpu" else gs.constants.backend.cpu
    try:
        gs.init(logging_level="warning", backend=backend)
        print(f"Genesis initialized with {args.backend} backend")
    except Exception as e:
        print(f"Warning: Genesis initialization issue: {e}")
        print("Continuing anyway...")

    # Get configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_genesis_configs()

    # Setup log directory
    log_dir = PATHS.tensorboard_log / args.exp_name
    if log_dir.exists():
        print(f"Warning: Log directory {log_dir} already exists. Contents will be overwritten.")
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    print(f"Creating Spot Genesis Environment with {args.num_envs} instances...")
    print(f"Using device: {args.device}")
    
    env_base = SpotGenesisEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
        device=args.device,
    )

    # Wrap for Stable Baselines 3
    print("Wrapping environment for SB3 compatibility...")
    env = GenesisSB3Wrapper(env_base)

    # Setup PPO with project config
    policy_kwargs = TRAINING.policy_kwargs.copy() if TRAINING.policy_kwargs else {}
    if 'activation_fn' in policy_kwargs and isinstance(policy_kwargs['activation_fn'], str):
        import torch.nn as nn
        activation_name = policy_kwargs.pop('activation_fn')
        if activation_name == 'elu':
            policy_kwargs['activation_fn'] = nn.ELU
        else:
            policy_kwargs['activation_fn'] = nn.ReLU

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = TRAINING.batch_size if TRAINING.batch_size > 0 else TRAINING.n_steps // 4

    # Use GPU for PPO if available and device is CUDA
    ppo_device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    print(f"PPO will use device: {ppo_device}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=TRAINING.learning_rate,
        n_steps=TRAINING.n_steps,
        batch_size=batch_size,
        n_epochs=TRAINING.n_epochs,
        gamma=TRAINING.gamma,
        gae_lambda=TRAINING.gae_lambda,
        clip_range=TRAINING.clip_range,
        ent_coef=TRAINING.ent_coef,
        vf_coef=TRAINING.vf_coef,
        max_grad_norm=TRAINING.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        device=ppo_device,
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=str(log_dir),
        name_prefix="spot_ppo",
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training with {args.num_envs} environments")
    print(f"Total timesteps: {args.max_steps:,}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    model.learn(total_timesteps=args.max_steps, callback=checkpoint_callback)

    # Save final model
    final_model_path = log_dir / "final_model"
    print(f"\nTraining complete. Saving final model to {final_model_path}")
    model.save(str(final_model_path))

    # Also save to project's model directory for compatibility
    project_model_path = PATHS.model_path.parent / f"ppo_spot_genesis.zip"
    print(f"Saving copy to {project_model_path}")
    model.save(str(project_model_path))

    env.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()

