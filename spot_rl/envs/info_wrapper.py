from __future__ import annotations

from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class InfoCollectorWrapper(gym.Wrapper):
    """Wrapper to collect info dicts for TensorBoard logging.
    
    This wrapper stores the info dicts from each step so they can be
    accessed by callbacks even when using VecNormalize.
    """
    
    def __init__(self, env: gym.Env, max_size: int = 10000):
        super().__init__(env)
        self.info_history: deque = deque(maxlen=max_size)
        self.last_info: Optional[dict] = None
    
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment and store the info dict."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = info.copy() if isinstance(info, dict) else {}
        self.info_history.append(self.last_info)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[Any, dict]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.last_info = info.copy() if isinstance(info, dict) else {}
        return obs, info
    
    def get_last_info(self) -> Optional[dict]:
        """Get the last info dict."""
        return self.last_info
    
    def get_info_history(self) -> list[dict]:
        """Get all collected info dicts."""
        return list(self.info_history)

