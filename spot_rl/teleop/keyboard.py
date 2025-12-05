from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np

from spot_rl.config import COMMAND

try:
    from pynput import keyboard

    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    keyboard = cast(Any, None)


@dataclass
class TeleopState:
    lin_x: float = 0.0
    lin_y: float = 0.0
    ang_z: float = 0.0
    height: float = 0.35
    jump: float = 0.0
    stop: bool = False

    def clamp(self):
        self.lin_x = float(np.clip(self.lin_x, *COMMAND.lin_vel_x_range))
        self.lin_y = float(np.clip(self.lin_y, *COMMAND.lin_vel_y_range))
        self.ang_z = float(np.clip(self.ang_z, *COMMAND.ang_vel_range))
        self.height = float(np.clip(self.height, *COMMAND.height_range))
        self.jump = float(np.clip(self.jump, *COMMAND.jump_range))

    def print_status(self):
        os.system("clear" if os.name != "nt" else "cls")
        print("--- Robot Teleop Commands ---")
        print(f"Forward (w/s): {self.lin_x:.2f} m/s")
        print(f"Strafe (a/d):  {self.lin_y:.2f} m/s")
        print(f"Turn (q/e):    {self.ang_z:.2f} rad/s")
        print(f"Height (i/k):   {self.height:.2f} m")
        print(f"Jump (j):       {self.jump:.2f} m")
        print("\nPress '8' to stop, 'r' to reset jump.")


class KeyboardController:
    def __init__(self, state: Optional[TeleopState] = None):
        if not HAS_PYNPUT:
            raise ImportError(
                "pynput is not installed. Install it via `pip install pynput` "
                "or disable teleoperation."
            )

        self.state = state or TeleopState()
        self.listener: Optional[keyboard.Listener] = None

    def _on_press(self, key):
        try:
            if key.char == "w":
                self.state.lin_x += 0.1
            elif key.char == "s":
                self.state.lin_x -= 0.1
            elif key.char == "a":
                self.state.lin_y += 0.1
            elif key.char == "d":
                self.state.lin_y -= 0.1
            elif key.char == "q":
                self.state.ang_z += 0.1
            elif key.char == "e":
                self.state.ang_z -= 0.1
            elif key.char == "i":
                self.state.height += 0.02
            elif key.char == "k":
                self.state.height -= 0.02
            elif key.char == "j":
                self.state.jump = COMMAND.jump_range[1]
            elif key.char == "r":
                self.state.jump = 0.0
            elif key.char == "8":
                self.state.stop = True
        except AttributeError:
            if key == keyboard.Key.space:
                self.state.jump = COMMAND.jump_range[1]
            return

        self.state.clamp()
        self.state.print_status()

    def _on_release(self, key):
        if key == keyboard.Key.esc:
            return False
        return None

    def start(self):
        if self.listener is not None:
            return

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.state.print_status()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None


