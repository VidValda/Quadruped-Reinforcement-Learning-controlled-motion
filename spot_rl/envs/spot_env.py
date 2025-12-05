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
    env = gym.wrappers.TimeLimit(env, max_episode_steps=SIMULATION.max_episode_steps)
    return env


