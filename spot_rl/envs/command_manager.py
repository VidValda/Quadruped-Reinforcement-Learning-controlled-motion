from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: Tuple[float, float]
    lin_vel_y_range: Tuple[float, float]
    ang_vel_range: Tuple[float, float]
    height_range: Tuple[float, float]
    jump_range: Tuple[float, float]
    resampling_time_s: float
    num_commands: int = 5


def rand_float(lower, upper, size):
    return (upper - lower) * np.random.random(size) + lower


def rand_gaussian(mean, min_val, max_val, n_std, size):
    mean_arr = np.full(size, mean)
    std = (max_val - min_val) / 4.0 * n_std
    return np.clip(np.random.normal(mean_arr, std), min_val, max_val)


class CommandManager:

    def __init__(self, config: CommandConfig, dt: float, reward_cfg=None) -> None:
        self._config = config
        self._manual_control = False
        self._np_random = None
        self.reward_cfg = reward_cfg

        self.resampling_steps = max(1, int(self._config.resampling_time_s / dt))
        self.steps_since_resample = 0

        self.commands = np.zeros(self._config.num_commands)
        self.last_actions = None

    @property
    def manual_control(self) -> bool:
        return self._manual_control

    @property
    def target_lin_vel(self):
        return self.commands[:2]

    @property
    def target_ang_vel(self):
        return self.commands[2]

    def bind_random_generator(self, np_random: np.random.Generator) -> None:
        self._np_random = np_random

    def enable_manual_control(self) -> None:
        self._manual_control = True

    def disable_manual_control(self) -> None:
        self._manual_control = False

    def set_manual_targets(self, lin_vel, ang_vel: float) -> None:
        self.commands[0] = lin_vel[0]
        self.commands[1] = lin_vel[1]
        self.commands[2] = ang_vel

    def reset(self) -> None:
        self.steps_since_resample = 0
        if not self.manual_control:
            self._sample_commands()

    def step(self, is_train=True) -> None:
        self.steps_since_resample += 1
        if self.manual_control:
            return

        if self.steps_since_resample % self.resampling_steps == 0:
            if is_train:
                self._sample_commands()

    def _resample_commands(self):
        if self._np_random is None:
            raise RuntimeError("CommandManager requires a bound random generator before sampling.")

        self.commands[0] = rand_gaussian(
            self.commands[0],
            *self._config.lin_vel_x_range,
            2.0,
            1
        )[0]
        self.commands[1] = rand_gaussian(
            self.commands[1],
            *self._config.lin_vel_y_range,
            2.0,
            1
        )[0]
        self.commands[2] = rand_gaussian(
            self.commands[2],
            *self._config.ang_vel_range,
            2.0,
            1
        )[0]
        self.commands[3] = rand_gaussian(
            self.commands[3],
            *self._config.height_range,
            0.5,
            1
        )[0]
        self.commands[4] = 0.0

        if self.reward_cfg:
            height_diff_scale = 0.5 + abs(self.commands[3] - self.reward_cfg["base_height_target"]) / (
                self._config.height_range[1] - self.reward_cfg["base_height_target"]
            ) * 0.5
            self.commands[0] *= height_diff_scale
            self.commands[1] *= height_diff_scale
            self.commands[2] *= height_diff_scale

    def _sample_commands(self):
        if self._np_random is None:
            raise RuntimeError("CommandManager requires a bound random generator before sampling.")

        self.commands[0] = self._np_random.uniform(*self._config.lin_vel_x_range)
        self.commands[1] = self._np_random.uniform(*self._config.lin_vel_y_range)
        self.commands[2] = self._np_random.uniform(*self._config.ang_vel_range)
        self.commands[3] = self._np_random.uniform(*self._config.height_range)
        self.commands[4] = 0.0

        if self.reward_cfg:
            height_diff_scale = 0.5 + abs(self.commands[3] - self.reward_cfg["base_height_target"]) / (
                self._config.height_range[1] - self.reward_cfg["base_height_target"]
            ) * 0.5
            self.commands[0] *= height_diff_scale
            self.commands[1] *= height_diff_scale
            self.commands[2] *= height_diff_scale

    def _sample_jump_commands(self):
        if self._np_random is None:
            raise RuntimeError("CommandManager requires a bound random generator before sampling.")

        self.commands[4] = self._np_random.uniform(*self._config.jump_range)

    def update_last_actions(self, actions):
        if self.last_actions is None:
            self.last_actions = np.zeros(len(actions))
        self.last_actions[:] = actions[:]
