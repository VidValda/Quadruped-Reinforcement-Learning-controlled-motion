from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from spot_rl.config import COMMAND, SIMULATION
from spot_rl.envs.command_manager import CommandConfig, CommandManager
from spot_rl.envs.model_loader import SpotModelLoader
from spot_rl.envs.observation_builder import build_observation
from spot_rl.envs.reward_calculator import SpotRewardCalculator


class CustomSpotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.model = SpotModelLoader().build()
        self.data = mujoco.MjData(self.model)

        self.frame_skip = SIMULATION.frame_skip
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

        self.target_height = SIMULATION.target_height
        self.last_action = np.zeros(self.model.nu)

        command_config = CommandConfig(
            lin_vel_x_range=COMMAND.lin_vel_x_range,
            lin_vel_y_range=COMMAND.lin_vel_y_range,
            ang_vel_range=COMMAND.ang_vel_range,
            resampling_time_s=COMMAND.resampling_time_s,
        )
        self.command_manager = CommandManager(command_config, self.dt)

        self.reward_calculator = SpotRewardCalculator(
            target_height=self.target_height,
            model=self.model,
            default_homing_pose=self.default_homing_pose,
            dt=self.dt
        )
        
        # Domain randomization
        self.friction_range = (0.5, 1.25)
        self.push_interval = 10.0  # seconds
        self.push_magnitude = 0.5  # m/s
        self.last_push_time = 0.0
        self.current_friction = None

        num_actuators = self.model.nu
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(num_actuators,), dtype=np.float32)

        num_joint_pos = self.model.nq - 7
        num_joint_vel = self.model.nv - 6
        num_root_vel = 6
        num_sensors = 0
        num_commands = 3

        # Base observation dimension (single frame)
        base_obs_dim = num_joint_pos + num_joint_vel + num_root_vel + 1 + 2 + num_sensors + num_commands
        
        # Frame stacking: 3 frames (current, t-1, t-2)
        self.num_frames = 3
        total_obs_dim = base_obs_dim * self.num_frames
        
        # Store observation history for frame stacking
        self.obs_history = [np.zeros(base_obs_dim, dtype=np.float32) for _ in range(self.num_frames)]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

    def _get_obs(self):
        """Get current observation and stack with previous frames."""
        current_obs = build_observation(
            self.data,
            self.torso_body_id,
            self.command_manager.target_lin_vel,
            self.command_manager.target_ang_vel,
        )
        
        # Update observation history: shift frames and add current
        # obs_history[0] = t-2, obs_history[1] = t-1, obs_history[2] = current
        self.obs_history = self.obs_history[1:] + [current_obs]
        
        # Stack all frames: [current, t-1, t-2]
        stacked_obs = np.concatenate(self.obs_history)
        return stacked_obs

    def enable_manual_control(self):
        self.command_manager.enable_manual_control()
        print("Manual control enabled in Environment.")

    def set_target_velocities(self, lin_vel, ang_vel):
        self.command_manager.set_manual_targets(lin_vel, ang_vel)

    @property
    def manual_control(self):
        return self.command_manager.manual_control

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.command_manager.bind_random_generator(self.np_random)
        mujoco.mj_resetData(self.model, self.data)

        initial_pos = SIMULATION.initial_position
        self.data.qpos[0] = initial_pos[0]  # x
        self.data.qpos[1] = initial_pos[1]  # y
        self.data.qpos[2] = initial_pos[2]  # z

        # Initialize robot in homing pose
        if len(self.default_homing_pose) == self.model.nu:
            self.data.qpos[7:] = self.default_homing_pose
            # Set control inputs to match homing pose for consistency
            self.data.ctrl[:] = self.default_homing_pose
            mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.model.nu)
        self.command_manager.reset()
        
        # Domain randomization: Randomize friction every episode
        self._randomize_friction()
        self.last_push_time = 0.0
        
        # Reset observation history (fill with current observation)
        base_obs = build_observation(
            self.data,
            self.torso_body_id,
            self.command_manager.target_lin_vel,
            self.command_manager.target_ang_vel,
        )
        self.obs_history = [base_obs.copy() for _ in range(self.num_frames)]

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info
    
    def _randomize_friction(self):
        """Randomize ground friction between [0.5, 1.25] every episode."""
        friction_min, friction_max = self.friction_range
        self.current_friction = self.np_random.uniform(friction_min, friction_max)
        
        # Apply friction to all geoms that contact the floor
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_geom_id != -1:
            # Set friction for floor geom
            self.model.geom_friction[floor_geom_id, 0] = self.current_friction
            self.model.geom_friction[floor_geom_id, 1] = self.current_friction
            self.model.geom_friction[floor_geom_id, 2] = self.current_friction
    
    def _apply_random_push(self, current_time: float):
        """Apply a random velocity shove to the robot's base every ~10 seconds."""
        if current_time - self.last_push_time >= self.push_interval:
            # Random direction in xy plane
            angle = self.np_random.uniform(0, 2 * np.pi)
            push_vel_x = self.push_magnitude * np.cos(angle)
            push_vel_y = self.push_magnitude * np.sin(angle)
            
            # Apply push to base velocity
            self.data.qvel[0] += push_vel_x  # vx
            self.data.qvel[1] += push_vel_y  # vy
            
            self.last_push_time = current_time

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        final_action = self.default_homing_pose + action

        final_action_clipped = np.clip(final_action, -2 * np.pi, 2 * np.pi)
        self.data.ctrl[:] = final_action_clipped

        # Apply random push before step (domain randomization)
        current_time = self.data.time
        self._apply_random_push(current_time)

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
    env = gym.wrappers.TimeLimit(env, max_episode_steps=SIMULATION.max_episode_steps)
    return env


