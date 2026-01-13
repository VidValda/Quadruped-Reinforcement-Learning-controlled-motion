"""
Standalone training script for Google Colab / Kaggle with GPU support using MuJoCo MJX.
MJX is a JAX-based GPU-accelerated version of MuJoCo that significantly speeds up environment simulation.

Usage in Colab:
    !python train_colab_mjx.py --total_timesteps 10000000 --model_name ppo_spot_mjx

Usage in Kaggle:
    python train_colab_mjx.py --total_timesteps 10000000 --model_name ppo_spot_mjx
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
import jax
import jax.numpy as jnp
import mujoco
try:
    import mujoco.mjx as mjx
except ImportError:
    # Try alternative import
    try:
        from mujoco import mjx
    except ImportError:
        raise ImportError(
            "MJX not found. Install with: pip install mujoco[mjx] or pip install mujoco-mjx"
        )
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
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
    """Auto-detect GPU availability for JAX."""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        if gpu_devices:
            print(f"✓ GPU detected: {gpu_devices[0]}")
            print(f"✓ Available JAX devices: {devices}")
            return "gpu"
        else:
            print(f"⚠ GPU not available, using {devices[0]}")
            return "cpu"
    except Exception as e:
        print(f"⚠ Error detecting device: {e}, using CPU")
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

    def build_mj_model(self) -> mujoco.MjModel:
        """Build regular MuJoCo model for loading XML."""
        xml_string = self._load_mjcf()
        xml_string = self._ensure_mesh_paths(xml_string)
        xml_string = self._ensure_floor(xml_string)

        try:
            return mujoco.MjModel.from_xml_string(xml_string)
        except (mujoco.Error, RuntimeError, ValueError) as exc:
            print(f"Error compiling XML: {exc}. Loading default.")
            from robot_descriptions.loaders.mujoco import load_robot_description
            return load_robot_description("spot_mj_description")

    def build_mjx_model(self) -> mjx.Model:
        """Build MJX model from MuJoCo model."""
        mj_model = self.build_mj_model()
        return mjx.put_model(mj_model)

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
# Observation Builder (MJX version)
# ============================================================================


def build_observation_mjx(data: mjx.Data, mj_model: mujoco.MjModel, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    """Build observation from MJX data."""
    # Get body position and quaternion (keep in JAX as long as possible)
    torso_xpos = data.xpos[torso_body_id]
    torso_quat = data.xquat[torso_body_id]
    torso_z_pos = jnp.array(torso_xpos[2])
    
    # Convert quaternion to numpy for roll/pitch calculation (quat_to_roll_pitch uses numpy)
    quat_np = np.array(torso_quat)
    roll, pitch = quat_to_roll_pitch(quat_np)
    pitch_roll = jnp.array([pitch, roll])

    # Build observation in JAX, then convert to numpy once at the end
    obs = jnp.concatenate([
        data.qpos[7:],
        data.qvel[6:],
        data.qvel[0:6],
        jnp.array([torso_z_pos]),
        pitch_roll,
        jnp.array(target_lin_vel),
        jnp.array([target_ang_vel]),
    ])
    
    # Single GPU→CPU transfer
    return jax.device_get(obs).astype(np.float32)


# ============================================================================
# Reward Calculator (MJX version)
# ============================================================================


class SpotRewardCalculatorMJX:
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

    def __call__(self, data: mjx.Data, mj_model: mujoco.MjModel, action, last_action, target_lin_vel, target_ang_vel, torso_body_id: int):
        """Calculate reward from MJX data."""
        # Get body velocities and position
        # Use cvel if available, otherwise fall back to xvel
        if hasattr(data, 'cvel') and data.cvel.shape[0] > torso_body_id:
            torso_cvel = data.cvel[torso_body_id]
            current_lin_vel = np.array(torso_cvel[3:5])  # Linear velocity in x, y
            current_ang_vel = float(torso_cvel[2])  # Angular velocity around z
        else:
            # Fallback: compute from xvel
            torso_xvel = data.xvel[torso_body_id]
            current_lin_vel = np.array(torso_xvel[3:5])  # Linear velocity in x, y
            current_ang_vel = float(torso_xvel[2])  # Angular velocity around z
        
        torso_xpos = data.xpos[torso_body_id]
        torso_z_pos = float(torso_xpos[2])
        torso_quat = data.xquat[torso_body_id]
        
        # Convert to numpy for calculations
        quat_np = np.array(torso_quat)

        lin_vel_error = np.linalg.norm(target_lin_vel - current_lin_vel)
        ang_vel_error = np.square(target_ang_vel - current_ang_vel)

        lin_vel_reward = np.exp(-1.5 * lin_vel_error)
        ang_vel_reward = np.exp(-1.0 * ang_vel_error)

        roll, pitch = quat_to_roll_pitch(quat_np)

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
# Environment (MJX version)
# ============================================================================


class CustomSpotEnvMJX(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # Load models
        loader = SpotModelLoader()
        self.mj_model = loader.build_mj_model()  # Regular MuJoCo model for reference
        self.mjx_model = loader.build_mjx_model()  # MJX model for GPU computation
        self.data = mjx.make_data(self.mjx_model)
        
        # JIT compile the step function for performance
        # Note: We create a closure to capture the model
        def step_fn(data):
            return mjx.step(self.mjx_model, data)
        self.jit_step = jax.jit(step_fn)

        self.frame_skip = SimulationConfig.frame_skip
        self.dt = self.frame_skip * self.mj_model.opt.timestep
        self.render_mode = render_mode
        self.viewer = None

        self.torso_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "body")
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
            ],
            dtype=np.float32
        )

        self.target_height = SimulationConfig.target_height
        self.last_action = np.zeros(self.mj_model.nu, dtype=np.float32)

        command_config = CommandConfig(
            lin_vel_x_range=CommandConfig.lin_vel_x_range,
            lin_vel_y_range=CommandConfig.lin_vel_y_range,
            ang_vel_range=CommandConfig.ang_vel_range,
            resampling_time_s=CommandConfig.resampling_time_s,
        )
        self.command_manager = CommandManager(command_config, self.dt)

        self.reward_calculator = SpotRewardCalculatorMJX(target_height=self.target_height)

        num_actuators = self.mj_model.nu
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(num_actuators,), dtype=np.float32)

        num_joint_pos = self.mj_model.nq - 7
        num_joint_vel = self.mj_model.nv - 6
        num_root_vel = 6
        num_sensors = 0
        num_commands = 3

        total_obs_dim = num_joint_pos + num_joint_vel + num_root_vel + 1 + 2 + num_sensors + num_commands

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

    def _get_obs(self):
        return build_observation_mjx(
            self.data,
            self.mj_model,
            self.torso_body_id,
            self.command_manager.target_lin_vel,
            self.command_manager.target_ang_vel,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.command_manager.bind_random_generator(self.np_random)
        
        # Reset MJX data
        self.data = mjx.make_data(self.mjx_model)

        # Initialize robot in homing pose
        if len(self.default_homing_pose) == self.mj_model.nu:
            qpos = np.array(self.data.qpos)
            # Ensure root position is at reasonable height
            qpos[2] = self.target_height  # z position
            # Set joint positions
            qpos[7:] = self.default_homing_pose
            self.data = self.data.replace(qpos=jnp.array(qpos))
            
            # Set control inputs to match homing pose
            ctrl = jnp.array(self.default_homing_pose)
            self.data = self.data.replace(ctrl=ctrl)
            
            # Forward kinematics
            self.data = mjx.forward(self.mjx_model, self.data)

        self.last_action = np.zeros(self.mj_model.nu, dtype=np.float32)
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
        
        # Set control
        self.data = self.data.replace(ctrl=jnp.array(final_action_clipped))

        # Step simulation with frame_skip (using JIT-compiled step)
        for _ in range(self.frame_skip):
            self.data = self.jit_step(self.data)
        
        self.command_manager.step()

        obs = self._get_obs()

        reward, terminated, info = self.reward_calculator(
            self.data,
            self.mj_model,
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
            # For rendering, we need to use regular MuJoCo
            mj_data = mujoco.MjData(self.mj_model)
            mj_data.qpos[:] = np.array(self.data.qpos)
            mj_data.qvel[:] = np.array(self.data.qvel)
            mujoco.mj_forward(self.mj_model, mj_data)
            
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.mj_model, mj_data)

        # Update viewer with current state
        mj_data = mujoco.MjData(self.mj_model)
        mj_data.qpos[:] = np.array(self.data.qpos)
        mj_data.qvel[:] = np.array(self.data.qvel)
        mujoco.mj_forward(self.mj_model, mj_data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_env(render_mode: Optional[str] = None):
    """Create environment with proper JAX initialization."""
    # Ensure JAX is properly initialized (important for multiprocessing)
    # This is a no-op if already initialized, but ensures subprocesses have JAX ready
    try:
        _ = jax.devices()
    except Exception:
        pass  # JAX will initialize on first use
    
    env = CustomSpotEnvMJX(render_mode=render_mode)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=SimulationConfig.max_episode_steps)
    return env


# ============================================================================
# Training Pipeline
# ============================================================================


def build_training_env(num_envs: Optional[int] = None) -> VecNormalize:
    """Build vectorized training environment."""
    if num_envs is None:
        num_envs = multiprocessing.cpu_count()
    
    # Detect if we're in Colab/Kaggle
    in_colab = os.path.exists("/content") or os.getenv("COLAB_GPU") is not None
    in_kaggle = os.path.exists("/kaggle") or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None
    
    # Use DummyVecEnv for single environment or when in Colab/Kaggle (JAX multiprocessing issues)
    # SubprocVecEnv can cause deadlocks with JAX/MJX in cloud environments
    # Also use DummyVecEnv if num_envs is 1 (no benefit from multiprocessing)
    use_subproc = num_envs > 1 and not in_colab and not in_kaggle
    
    if use_subproc:
        print(f"Creating {num_envs} parallel environments with MJX (GPU-accelerated) using SubprocVecEnv...")
        def make_env_fn():
            return make_env(render_mode=None)
        env_fns = [make_env_fn for _ in range(num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        vec_env_type = "DummyVecEnv"
        if in_colab:
            vec_env_type += " (Colab detected)"
        elif in_kaggle:
            vec_env_type += " (Kaggle detected)"
        elif num_envs == 1:
            vec_env_type += " (single environment)"
        print(f"Creating {num_envs} parallel environments with MJX (GPU-accelerated) using {vec_env_type}...")
        def make_env_fn():
            return make_env(render_mode=None)
        env_fns = [make_env_fn for _ in range(num_envs)]
        env = DummyVecEnv(env_fns)
    
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
    print("SPOT ROBOT REINFORCEMENT LEARNING TRAINING (MJX GPU)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PPO device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print(f"Training config: n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, lr={learning_rate}")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if tensorboard_log:
        os.makedirs(tensorboard_log, exist_ok=True)
    
    # Build environment
    env = build_training_env(num_envs=num_envs)
    
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
    print("\nStarting training with GPU-accelerated MJX environments...")
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
        description="Train Spot robot using PPO with MJX GPU acceleration for Colab/Kaggle"
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
        default="ppo_spot_mjx",
        help="Name for saved model files (default: ppo_spot_mjx)",
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
        help="Device for PPO: 'cuda', 'cpu', or 'auto' (default: auto-detect). Note: MJX runs on GPU automatically.",
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
    
    # Detect device for PPO (separate from MJX which uses JAX/GPU)
    if args.device is None or args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✓ PPO will use GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("⚠ PPO will use CPU (MJX environments still run on GPU)")
        except ImportError:
            device = "cpu"
            print("⚠ PyTorch not available, PPO will use CPU (MJX environments still run on GPU)")
    else:
        device = args.device
        print(f"Using PPO device: {device}")
    
    # Check JAX/MJX setup
    detect_device()
    
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

