from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardMetricsCallback(BaseCallback):
    """Callback to log custom metrics from environment info dict to TensorBoard.
    
    This callback extracts metrics from the info dict returned by the environment
    and logs them to TensorBoard for visualization. Metrics are averaged over
    each rollout period.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        """Called at each step. Extract metrics from info dict."""
        # In _on_step, we can access self.locals which contains the current step info
        # However, for vectorized envs, it's better to collect in _on_rollout_end
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout. Extract and log metrics from rollout infos."""
        # Access rollout data from locals
        # In stable-baselines3, rollout data can be in different places
        infos = None
        
        # Try different ways to access infos
        if hasattr(self.locals, 'infos'):
            infos = self.locals.infos
        elif isinstance(self.locals, dict) and 'infos' in self.locals:
            infos = self.locals['infos']
        elif hasattr(self, 'locals') and isinstance(self.locals, dict):
            infos = self.locals.get('infos', None)
        
        if infos is None:
            return
        
        metrics = defaultdict(list)
        
        # List of metric keys to extract and log
        metric_keys = [
            "rewards/lin_vel",
            "rewards/ang_vel",
            "rewards/orientation",
            "rewards/torques",
            "rewards/action_rate",
            "tracking/linear_velocity_error",
            "tracking/angular_velocity_error",
            "performance/action_rate",
        ]
        
        # Process infos - can be a list of dicts (vectorized) or a single dict
        # In vectorized envs, infos is typically a list of lists (one per step, each containing dicts per env)
        if isinstance(infos, list):
            # Flatten if it's a list of lists (rollout steps)
            all_infos = []
            for item in infos:
                if isinstance(item, list):
                    all_infos.extend(item)
                elif isinstance(item, dict):
                    all_infos.append(item)
            
            # Process all collected infos
            for info in all_infos:
                if isinstance(info, dict):
                    for key in metric_keys:
                        if key in info:
                            value = info[key]
                            # Convert to float if it's a numpy type
                            if isinstance(value, (np.ndarray, np.generic)):
                                value = float(value)
                            elif isinstance(value, (int, float)):
                                value = float(value)
                            else:
                                continue  # Skip if not a numeric type
                            metrics[key].append(value)
        elif isinstance(infos, dict):
            # Single environment: infos is a single dict
            for key in metric_keys:
                if key in infos:
                    value = infos[key]
                    if isinstance(value, (np.ndarray, np.generic)):
                        value = float(value)
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    else:
                        continue
                    metrics[key].append(value)
        
        # Calculate mean for each metric and log to TensorBoard
        if metrics and self.logger is not None:
            for metric_name, values in metrics.items():
                if values:
                    mean_value = np.mean(values)
                    self.logger.record(metric_name, mean_value)
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Log any remaining metrics
        self._on_rollout_end()

