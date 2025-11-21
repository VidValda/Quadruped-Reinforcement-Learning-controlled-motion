from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: Tuple[float, float]
    lin_vel_y_range: Tuple[float, float]
    ang_vel_range: Tuple[float, float]
    resampling_time_s: float


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

