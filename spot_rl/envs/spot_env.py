import math
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from spot_rl.config import COMMAND, SIMULATION, REWARD, OBS
from spot_rl.envs.command_manager import CommandConfig, CommandManager
from spot_rl.envs.model_loader import SpotModelLoader
from spot_rl.envs.observation_builder import build_observation
from spot_rl.envs.reward_calculator import SpotRewardCalculator
from spot_rl.envs.utils import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


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

        self.simulate_action_latency = SIMULATION.simulate_action_latency
        self.max_episode_length = math.ceil(SIMULATION.episode_length_s / self.dt)

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

        self.base_init_pos = np.array([0.0, 0.0, SIMULATION.target_height])
        self.base_init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        command_config = CommandConfig(
            lin_vel_x_range=COMMAND.lin_vel_x_range,
            lin_vel_y_range=COMMAND.lin_vel_y_range,
            ang_vel_range=COMMAND.ang_vel_range,
            height_range=COMMAND.height_range,
            jump_range=COMMAND.jump_range,
            resampling_time_s=COMMAND.resampling_time_s,
            num_commands=COMMAND.num_commands,
        )
        reward_cfg_dict = {
            "base_height_target": REWARD.base_height_target,
        }
        self.command_manager = CommandManager(command_config, self.dt, reward_cfg_dict)

        self.reward_calculator = SpotRewardCalculator(
            target_height=REWARD.base_height_target,
            reward_scales=REWARD.reward_scales,
            dt=self.dt,
            jump_reward_steps=REWARD.jump_reward_steps,
            tracking_sigma=REWARD.tracking_sigma,
        )

        self.obs_scales = OBS.obs_scales
        self.reward_scales = REWARD.reward_scales

        num_actuators = self.model.nu
        self.action_space = spaces.Box(
            low=-SIMULATION.clip_actions,
            high=SIMULATION.clip_actions,
            shape=(num_actuators,),
            dtype=np.float32
        )

        self.num_obs = OBS.num_obs
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32
        )

        self.commands_scale = np.array([
            self.obs_scales["lin_vel"],
            self.obs_scales["lin_vel"],
            self.obs_scales["ang_vel"],
            self.obs_scales["lin_vel"],
            self.obs_scales["lin_vel"],
        ])

        self.actions = np.zeros(num_actuators)
        self.last_actions = np.zeros(num_actuators)
        self.last_dof_vel = np.zeros(num_actuators)

        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.zeros(3)
        self.global_gravity = np.array([0.0, 0.0, -1.0])

        self.base_pos = np.zeros(3)
        self.base_quat = np.zeros(4)
        self.base_euler = np.zeros(3)

        self.dof_pos = np.zeros(num_actuators)
        self.dof_vel = np.zeros(num_actuators)

        self.episode_length = 0
        self.jump_toggled_buf = 0.0
        self.jump_target_height = 0.0

        self.extras = {}

    def _get_obs(self):
        return build_observation(
            self.data,
            self.torso_body_id,
            self.command_manager.commands,
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            self.default_homing_pose,
            self.actions,
            self.jump_toggled_buf,
            REWARD.jump_reward_steps,
            self.obs_scales,
            self.commands_scale,
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

        self.episode_length = 0
        self.last_actions[:] = 0.0
        self.last_dof_vel[:] = 0.0
        self.jump_toggled_buf = 0.0
        self.jump_target_height = 0.0

        self.dof_pos[:] = self.default_homing_pose
        self.dof_vel[:] = 0.0
        self.data.qpos[7:] = self.dof_pos
        self.data.qvel[6:] = 0.0

        self.base_pos[:] = self.base_init_pos
        self.base_quat[:] = self.base_init_quat
        body_xpos = self.data.body(self.torso_body_id).xpos
        body_xquat = self.data.body(self.torso_body_id).xquat
        body_xpos[:] = self.base_pos
        body_xquat[:] = self.base_quat

        self.base_lin_vel[:] = 0.0
        self.base_ang_vel[:] = 0.0

        self.command_manager.reset()
        self.command_manager.commands[3] = REWARD.base_height_target
        self.command_manager.last_actions = np.zeros(self.model.nu)

        self.reward_calculator.reset_episode_sums()

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action, is_train=True):
        self.actions = np.clip(action, self.action_space.low, self.action_space.high)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * SIMULATION.action_scale + self.default_homing_pose

        target_dof_pos_clipped = np.clip(target_dof_pos, -2 * np.pi, 2 * np.pi)
        self.data.ctrl[:] = target_dof_pos_clipped

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        self.episode_length += 1

        body_xpos = self.data.body(self.torso_body_id).xpos
        body_xquat = self.data.body(self.torso_body_id).xquat
        body_cvel = self.data.body(self.torso_body_id).cvel

        self.base_pos[:] = body_xpos
        self.base_quat[:] = body_xquat

        transformed_quat = transform_quat_by_quat(
            np.ones_like(self.base_quat) * self.inv_base_init_quat,
            self.base_quat
        )
        self.base_euler[:] = quat_to_xyz(transformed_quat)

        inv_base_quat = inv_quat(self.base_quat)
        lin_vel_world = body_cvel[3:6]
        ang_vel_world = body_cvel[0:3]
        self.base_lin_vel[:] = transform_by_quat(lin_vel_world, inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(ang_vel_world, inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.data.qpos[7:]
        self.dof_vel[:] = self.data.qvel[6:]

        envs_idx = []
        if self.episode_length % int(SIMULATION.episode_length_s / self.dt / 4) == 0:
            envs_idx = [0]

        if is_train:
            if len(envs_idx) > 0:
                self.command_manager._sample_commands()

            if self.np_random.random() < 0.05:
                self.command_manager._sample_commands()

            if self.np_random.random() < 0.05:
                self.command_manager._sample_jump_commands()

        jump_cmd_now = 1.0 if self.command_manager.commands[4] > 0.0 else 0.0
        toggle_mask = 1.0 if (self.jump_toggled_buf == 0.0) and (jump_cmd_now > 0.0) else 0.0
        self.jump_toggled_buf += toggle_mask * REWARD.jump_reward_steps
        self.jump_toggled_buf = max(0.0, self.jump_toggled_buf - 1.0)

        if jump_cmd_now > 0.0:
            self.jump_target_height = self.command_manager.commands[4]

        terminated = self.episode_length > self.max_episode_length
        terminated |= abs(self.base_euler[1]) > SIMULATION.termination_if_pitch_greater_than
        terminated |= abs(self.base_euler[0]) > SIMULATION.termination_if_roll_greater_than

        time_out = self.episode_length > self.max_episode_length
        self.extras["time_outs"] = 1.0 if time_out else 0.0

        if terminated:
            obs, info = self.reset()
            return obs, reward, terminated, truncated, info

        reward, reward_terminated, info = self.reward_calculator(
            self.data,
            self.actions,
            self.last_actions,
            self.command_manager.commands,
            self.base_lin_vel,
            self.base_ang_vel,
            self.base_pos,
            self.base_quat,
            self.dof_pos,
            self.dof_vel,
            self.default_homing_pose,
            self.jump_toggled_buf,
            self.jump_target_height,
            self.torso_body_id,
        )

        terminated = terminated or reward_terminated

        obs = self._get_obs()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.command_manager.update_last_actions(self.actions)

        self.command_manager.commands[4] = 0.0

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
