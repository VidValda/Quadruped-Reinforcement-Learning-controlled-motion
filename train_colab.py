"""
Standalone training script for Google Colab / Kaggle with GPU support.
This script bundles all necessary code and can run independently.

Usage in Colab:
    !python train_colab.py --total_timesteps 10000000 --model_name ppo_spot_colab

Usage in Kaggle:
    python train_colab.py --total_timesteps 10000000 --model_name ppo_spot_kaggle
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_range: tuple[float, float] = (-0.5, 0.5)
    resampling_time_s: float = 4.0


@dataclass(frozen=True)
class SimulationConfig:
    frame_skip: int = 5
    target_height: float = 0.35
    max_episode_steps: int = 2000


class TrainingConfig:
    total_timesteps: int = 30_000_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99


# ============================================================================
# Utility Functions
# ============================================================================


def quat_to_roll_pitch(quat):
    """Convert quaternion to roll and pitch angles."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Calculate roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Calculate pitch with safety check for arcsin domain
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    return roll, pitch


def detect_device():
    """Auto-detect GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return "cuda"
        else:
            print("⚠ GPU not available, using CPU")
            return "cpu"
    except ImportError:
        print("⚠ PyTorch not available, using CPU")
        return "cpu"


# ============================================================================
# Command Manager
# ============================================================================


class CommandManager:
    def __init__(self, config: CommandConfig, dt: float) -> None:
        self._config = config
        self._manual_control = False
        self._np_random = None

        self.resampling_steps = max(1, int(self._config.resampling_time_s / dt))
        self.steps_since_resample = 0

        self.target_lin_vel = np.zeros(2)
        self.target_ang_vel = 0.0

    @property
    def manual_control(self) -> bool:
        return self._manual_control

    def bind_random_generator(self, np_random: np.random.Generator) -> None:
        self._np_random = np_random

    def enable_manual_control(self) -> None:
        self._manual_control = True

    def disable_manual_control(self) -> None:
        self._manual_control = False

    def set_manual_targets(self, lin_vel, ang_vel: float) -> None:
        self.target_lin_vel[0] = lin_vel[0]
        self.target_lin_vel[1] = lin_vel[1]
        self.target_ang_vel = ang_vel

    def reset(self) -> None:
        self.steps_since_resample = 0
        if not self.manual_control:
            self._resample_commands()

    def step(self) -> None:
        self.steps_since_resample += 1
        if self.manual_control:
            return

        if self.steps_since_resample % self.resampling_steps == 0:
            self._resample_commands()

    def _resample_commands(self) -> None:
        if self._np_random is None:
            raise RuntimeError("CommandManager requires a bound random generator before sampling.")

        self.target_lin_vel[0] = self._np_random.uniform(*self._config.lin_vel_x_range)
        self.target_lin_vel[1] = self._np_random.uniform(*self._config.lin_vel_y_range)
        self.target_ang_vel = self._np_random.uniform(*self._config.ang_vel_range)


# ============================================================================
# Model Loader
# ============================================================================


class SpotModelLoader:
    FLOOR_ASSET = """
        <asset>
            <texture type="2d" name="grid" builtin="checker" width="512" height="512"
                     rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
        </asset>
    """

    FLOOR_GEOM = '    <geom name="floor" type="plane" size="10 10 0.1" material="grid"/>'

    def __init__(self, assets_subdir: str = "assets") -> None:
        self.assets_subdir = assets_subdir

    def build(self) -> mujoco.MjModel:
        xml_string = self._load_mjcf()
        xml_string = self._ensure_mesh_paths(xml_string)
        xml_string = self._ensure_floor(xml_string)

        try:
            return mujoco.MjModel.from_xml_string(xml_string)
        except (mujoco.Error, RuntimeError, ValueError) as exc:
            print(f"Error compiling XML: {exc}. Loading default.")
            from robot_descriptions.loaders.mujoco import load_robot_description
            return load_robot_description("spot_mj_description")

    @staticmethod
    def _load_mjcf() -> str:
        try:
            from robot_descriptions import spot_mj_description
            xml_path = Path(spot_mj_description.MJCF_PATH)
            with xml_path.open("r", encoding="utf-8") as file:
                return file.read()
        except (ValueError, AttributeError, ImportError, RuntimeError) as e:
            # Handle git branch issue (master vs main) or import errors
            error_str = str(e)
            error_type = type(e).__name__
            # Check for various git/repository errors
            git_related_keywords = [
                "refs/heads/master", "refs/heads/main", "Reference", 
                "spot_mj_description", "does not exist", "clone",
                "git", "repository", "hexsha", "dereference"
            ]
            if any(keyword in error_str for keyword in git_related_keywords):
                print(f"⚠ robot_descriptions issue detected ({error_type}): {error_str[:150]}...")
                print("Using direct download fallback...")
                return SpotModelLoader._load_mjcf_direct()
            else:
                # For other errors, try fallback anyway if it's an import/attribute error
                if error_type in ["ImportError", "AttributeError"]:
                    print(f"⚠ robot_descriptions import issue ({error_type}), using fallback...")
                    return SpotModelLoader._load_mjcf_direct()
                raise
    
    @staticmethod
    def _load_mjcf_direct() -> str:
        """Direct download fallback when robot_descriptions fails."""
        import urllib.request
        import zipfile
        import shutil
        
        # Use a persistent cache directory
        cache_dir = Path.home() / ".cache" / "mujoco_menagerie"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_dir = cache_dir / "mujoco_menagerie-main"
        spot_xml = extracted_dir / "spot" / "spot.xml"
        
        # Check if already cached
        if spot_xml.exists():
            print("✓ Using cached Spot XML")
            with spot_xml.open("r", encoding="utf-8") as file:
                return file.read()
        
        # URL to the spot XML file in mujoco_menagerie
        repo_url = "https://github.com/deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
        zip_path = cache_dir / "menagerie.zip"
        
        try:
            print("Downloading Spot XML from mujoco_menagerie...")
            urllib.request.urlretrieve(repo_url, zip_path)
            
            # Remove old extraction if exists
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            
            # Try the expected path first
            if spot_xml.exists():
                print("✓ Successfully downloaded and cached Spot XML")
                with spot_xml.open("r", encoding="utf-8") as file:
                    return file.read()
            
            # If not found, search recursively for spot.xml
            print(f"⚠ spot.xml not found at expected path, searching in {extracted_dir}...")
            spot_xml_files = list(extracted_dir.rglob("spot.xml"))
            
            if spot_xml_files:
                spot_xml = spot_xml_files[0]
                print(f"✓ Found spot.xml at: {spot_xml}")
                with spot_xml.open("r", encoding="utf-8") as file:
                    return file.read()
            
            # If still not found, try alternative: download directly from raw GitHub
            print("⚠ Trying direct download from GitHub raw...")
            raw_url = "https://raw.githubusercontent.com/deepmind/mujoco_menagerie/main/spot/spot.xml"
            spot_xml_direct = cache_dir / "spot.xml"
            try:
                urllib.request.urlretrieve(raw_url, spot_xml_direct)
                if spot_xml_direct.exists():
                    print("✓ Successfully downloaded Spot XML directly")
                    with spot_xml_direct.open("r", encoding="utf-8") as file:
                        return file.read()
            except Exception as direct_e:
                print(f"⚠ Direct download also failed: {direct_e}")
            
            # Last resort: list what's actually in the extracted directory
            if extracted_dir.exists():
                print(f"Contents of {extracted_dir}:")
                for item in extracted_dir.iterdir():
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir() and item.name == "spot":
                        print(f"    Contents of spot/:")
                        for subitem in item.iterdir():
                            print(f"      - {subitem.name}")
            
            raise FileNotFoundError(
                f"Could not find spot.xml in {extracted_dir}. "
                f"Searched recursively and tried direct download. "
                f"Please check your internet connection and the repository structure."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Spot XML: {e}. Please check your internet connection.")

    def _ensure_mesh_paths(self, xml_string: str) -> str:
        try:
            from robot_descriptions import spot_mj_description
            xml_path = Path(spot_mj_description.MJCF_PATH)
            xml_dir = xml_path.parent
        except (ValueError, AttributeError, ImportError):
            # If robot_descriptions failed, use our cache directory
            cache_dir = Path.home() / ".cache" / "mujoco_menagerie" / "mujoco_menagerie-main"
            xml_dir = cache_dir / "spot"
        
        assets_dir = os.path.abspath(xml_dir / self.assets_subdir).replace("\\", "/")

        if 'meshdir="assets"' in xml_string:
            return xml_string.replace('meshdir="assets"', f'meshdir="{assets_dir}"')

        if "<compiler" not in xml_string:
            return xml_string.replace(
                '<mujoco model="spot">',
                f'<mujoco model="spot">\n  <compiler meshdir="{assets_dir}"/>',
                1,
            )

        return xml_string.replace(
            "<compiler",
            f'<compiler meshdir="{assets_dir}"',
            1,
        )

    def _ensure_floor(self, xml_string: str) -> str:
        if "<asset>" not in xml_string:
            if "</compiler>" in xml_string:
                xml_string = xml_string.replace("</compiler>", f"</compiler>\n{self.FLOOR_ASSET}", 1)
            else:
                xml_string = xml_string.replace('<mujoco model="spot">', f'<mujoco model="spot">\n{self.FLOOR_ASSET}', 1)
        else:
            asset_additions = """
            <texture type="2d" name="grid" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
            """
            xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_additions}", 1)

        if "<worldbody>" in xml_string:
            return xml_string.replace("<worldbody>", f"<worldbody>\n{self.FLOOR_GEOM}", 1)

        print("Warning: <worldbody> not found. Floor not added.")
        return xml_string


# ============================================================================
# Observation Builder
# ============================================================================


def build_observation(data, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    torso_xpos = data.body(torso_body_id).xpos
    torso_quat = data.body(torso_body_id).xquat
    torso_z_pos = torso_xpos[2]
    
    roll, pitch = quat_to_roll_pitch(torso_quat)
    pitch_roll = np.array([pitch, roll])

    return np.concatenate(
        [
            data.qpos[7:],
            data.qvel[6:],
            data.qvel[0:6],
            np.array([torso_z_pos]),
            pitch_roll,
            target_lin_vel,
            np.array([target_ang_vel]),
        ]
    ).astype(np.float32)


# ============================================================================
# Reward Calculator
# ============================================================================


class SpotRewardCalculator:
    def __init__(
        self,
        target_height: float,
        lin_vel_weight: float = 2.0,
        ang_vel_weight: float = 1.0,
        height_penalty_weight: float = 2.0,
        orientation_penalty_weight: float = 1.0,
        action_rate_weight: float = 1,
        control_cost_weight: float = 0.03,
        termination_height_threshold: float = 0.2,
        termination_reward: float = -10.0,
    ) -> None:
        self.target_height = target_height
        self.lin_vel_weight = lin_vel_weight
        self.ang_vel_weight = ang_vel_weight
        self.height_penalty_weight = height_penalty_weight
        self.orientation_penalty_weight = orientation_penalty_weight
        self.action_rate_weight = action_rate_weight
        self.control_cost_weight = control_cost_weight
        self.termination_height_threshold = termination_height_threshold
        self.termination_reward = termination_reward

    def __call__(self, data, action, last_action, target_lin_vel, target_ang_vel, torso_body_id: int):
        current_lin_vel = data.body(torso_body_id).cvel[3:5]
        current_ang_vel = data.body(torso_body_id).cvel[2]
        torso_z_pos = data.body(torso_body_id).xpos[2]
        torso_quat = data.body(torso_body_id).xquat

        lin_vel_error = np.linalg.norm(target_lin_vel - current_lin_vel)
        ang_vel_error = np.square(target_ang_vel - current_ang_vel)

        lin_vel_reward = np.exp(-1.5 * lin_vel_error)
        ang_vel_reward = np.exp(-1.0 * ang_vel_error)

        roll, pitch = quat_to_roll_pitch(torso_quat)

        height_penalty = np.square(torso_z_pos - self.target_height)
        orientation_penalty = np.square(roll) + np.square(pitch)

        action_rate_penalty = np.sum(np.square(action - last_action))
        control_cost = np.sum(np.square(action))

        reward = (
            self.lin_vel_weight * lin_vel_reward
            + self.ang_vel_weight * ang_vel_reward
            - self.height_penalty_weight * height_penalty
            - self.orientation_penalty_weight * orientation_penalty
            - self.action_rate_weight * action_rate_penalty
            - self.control_cost_weight * control_cost
        )

        terminated = torso_z_pos < self.termination_height_threshold
        if terminated:
            reward = self.termination_reward

        info = {
            "lin_vel_error": float(lin_vel_error),
            "ang_vel_error": float(ang_vel_error),
            "torso_height": float(torso_z_pos),
            "roll": float(roll),
            "pitch": float(pitch),
        }

        return reward, terminated, info


# ============================================================================
# Environment
# ============================================================================


class CustomSpotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.model = SpotModelLoader().build()
        self.data = mujoco.MjData(self.model)

        self.frame_skip = SimulationConfig.frame_skip
        self.dt = self.frame_skip * self.model.opt.timestep
        self.render_mode = render_mode
        self.viewer = None

        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")
        if self.torso_body_id == -1:
            raise ValueError("Could not find body named 'body' in the XML model.")

        self.default_homing_pose = np.array(
            [
                0.0,
                0.7,
                -1.4,
                0.0,
                0.7,
                -1.4,
                0.0,
                0.7,
                -1.4,
                0.0,
                0.7,
                -1.4,
            ]
        )

        self.target_height = SimulationConfig.target_height
        self.last_action = np.zeros(self.model.nu)

        command_config = CommandConfig(
            lin_vel_x_range=CommandConfig.lin_vel_x_range,
            lin_vel_y_range=CommandConfig.lin_vel_y_range,
            ang_vel_range=CommandConfig.ang_vel_range,
            resampling_time_s=CommandConfig.resampling_time_s,
        )
        self.command_manager = CommandManager(command_config, self.dt)

        self.reward_calculator = SpotRewardCalculator(target_height=self.target_height)

        num_actuators = self.model.nu
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(num_actuators,), dtype=np.float32)

        num_joint_pos = self.model.nq - 7
        num_joint_vel = self.model.nv - 6
        num_root_vel = 6
        num_sensors = 0
        num_commands = 3

        total_obs_dim = num_joint_pos + num_joint_vel + num_root_vel + 1 + 2 + num_sensors + num_commands

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

    def _get_obs(self):
        return build_observation(
            self.data,
            self.torso_body_id,
            self.command_manager.target_lin_vel,
            self.command_manager.target_ang_vel,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.command_manager.bind_random_generator(self.np_random)
        mujoco.mj_resetData(self.model, self.data)

        # Initialize robot in homing pose
        if len(self.default_homing_pose) == self.model.nu:
            self.data.qpos[7:] = self.default_homing_pose
            # Set control inputs to match homing pose for consistency
            self.data.ctrl[:] = self.default_homing_pose
            mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.model.nu)
        self.command_manager.reset()

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        final_action = self.default_homing_pose + action

        final_action_clipped = np.clip(final_action, -2 * np.pi, 2 * np.pi)
        self.data.ctrl[:] = final_action_clipped

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.command_manager.step()

        obs = self._get_obs()

        reward, terminated, info = self.reward_calculator(
            self.data,
            action,
            self.last_action,
            self.command_manager.target_lin_vel,
            self.command_manager.target_ang_vel,
            self.torso_body_id,
        )
        self.last_action = action

        if terminated:
            info["termination_reason"] = "low_height"

        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_env(render_mode: Optional[str] = None):
    env = CustomSpotEnv(render_mode=render_mode)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=SimulationConfig.max_episode_steps)
    return env


# ============================================================================
# Training Pipeline
# ============================================================================


def build_training_env(num_envs: Optional[int] = None, tensorboard_log: Optional[str] = None) -> VecNormalize:
    """Build vectorized training environment."""
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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=TrainingConfig.gamma)
    return env


def create_model(
    env: VecNormalize,
    device: str,
    tensorboard_log: Optional[str] = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
) -> PPO:
    """Create PPO model with specified device."""
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=TrainingConfig.gamma,
        device=device,
    )


def train(
    total_timesteps: int,
    device: str,
    output_dir: str = "./models",
    model_name: str = "ppo_spot",
    tensorboard_log: Optional[str] = None,
    num_envs: Optional[int] = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
):
    """Main training function."""
    print("=" * 60)
    print("SPOT ROBOT REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print(f"Training config: n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, lr={learning_rate}")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if tensorboard_log:
        os.makedirs(tensorboard_log, exist_ok=True)
    
    # Build environment (with Monitor wrapper for TensorBoard metrics)
    env = build_training_env(num_envs=num_envs, tensorboard_log=tensorboard_log)
    
    # Create model
    model = create_model(
        env,
        device=device,
        tensorboard_log=tensorboard_log,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )
    
    # Train
    print("\nStarting training...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model and stats
    model_path = os.path.join(output_dir, f"{model_name}.zip")
    stats_path = os.path.join(output_dir, f"{model_name}_stats.pkl")
    
    print(f"\nSaving model to {model_path}")
    model.save(model_path)
    
    print(f"Saving normalization stats to {stats_path}")
    env.save(stats_path)
    
    env.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved: {model_path}")
    print(f"Stats saved: {stats_path}")
    print("=" * 60)
    
    return model, env


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Spot robot using PPO with GPU support for Colab/Kaggle"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=30_000_000,
        help="Total number of timesteps to train (default: 30,000,000)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ppo_spot_colab",
        help="Name for saved model files (default: ppo_spot_colab)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save models and stats (default: ./models)",
    )
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default=None,
        help="Directory for tensorboard logs (default: None, no logging)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: CPU count)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto-detect)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps per update (default: 2048)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs per update (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    
    args = parser.parse_args()
    
    # Detect device
    if args.device is None or args.device == "auto":
        device = detect_device()
    else:
        device = args.device
        print(f"Using device: {device}")
    
    # Train
    train(
        total_timesteps=args.total_timesteps,
        device=device,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tensorboard_log=args.tensorboard_log,
        num_envs=args.num_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

