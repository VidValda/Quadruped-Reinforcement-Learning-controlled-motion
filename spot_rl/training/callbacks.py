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
        # Dictionary to accumulate metrics during rollout
        self.metrics = defaultdict(list)
    
    def _on_step(self) -> bool:
        """Called at each step. Extract metrics from info dict."""
        # Access infos from locals - in vectorized envs, infos is a list of dicts
        infos = None
        
        # Try multiple ways to access infos
        if hasattr(self.locals, 'infos'):
            infos = self.locals.infos
        elif hasattr(self.locals, '__getitem__'):
            try:
                infos = self.locals['infos']
            except (KeyError, TypeError):
                pass
        
        # Also try accessing from the model's rollout buffer if available
        if infos is None and hasattr(self.model, 'rollout_buffer'):
            # Try to get infos from the rollout buffer
            if hasattr(self.model.rollout_buffer, 'infos'):
                infos = self.model.rollout_buffer.infos
        
        if infos is not None:
            self._extract_metrics_from_infos(infos)
        
        return True
    
    def _extract_metrics_from_infos(self, infos: Any) -> None:
        """Extract metrics from infos (can be list of dicts or single dict)."""
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
        
        # Handle different info formats
        if isinstance(infos, list):
            # Vectorized environment: list of info dicts (one per environment)
            for info in infos:
                if isinstance(info, dict):
                    for key in metric_keys:
                        if key in info:
                            value = info[key]
                            if isinstance(value, (np.ndarray, np.generic)):
                                value = float(value)
                            elif isinstance(value, (int, float)):
                                value = float(value)
                            else:
                                continue
                            self.metrics[key].append(value)
        elif isinstance(infos, dict):
            # Single environment: single info dict
            for key in metric_keys:
                if key in infos:
                    value = infos[key]
                    if isinstance(value, (np.ndarray, np.generic)):
                        value = float(value)
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    else:
                        continue
                    self.metrics[key].append(value)
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout. Log accumulated metrics to TensorBoard."""
        # Primary method: Get infos from rollout buffer (most reliable for vectorized envs)
        if hasattr(self.model, 'rollout_buffer') and hasattr(self.model.rollout_buffer, 'infos'):
            rollout_infos = self.model.rollout_buffer.infos
            if rollout_infos is not None and len(rollout_infos) > 0:
                # Rollout buffer infos is a list of lists: [step][env]
                # Each step contains a list of info dicts (one per environment)
                all_infos = []
                for step_infos in rollout_infos:
                    if isinstance(step_infos, list):
                        # Multiple environments
                        for env_info in step_infos:
                            if isinstance(env_info, dict):
                                all_infos.append(env_info)
                    elif isinstance(step_infos, dict):
                        # Single environment
                        all_infos.append(step_infos)
                
                # Extract metrics from all collected infos
                for info in all_infos:
                    if isinstance(info, dict):
                        self._extract_metrics_from_infos(info)
        
        # Also try to get infos from locals as fallback
        elif hasattr(self.locals, 'infos') and self.locals.infos is not None:
            self._extract_metrics_from_infos(self.locals.infos)
        
        # Log accumulated metrics from the rollout
        if self.metrics and self.logger is not None:
            for metric_name, values in self.metrics.items():
                if values:
                    mean_value = np.mean(values)
                    self.logger.record(metric_name, mean_value)
        
        # Reset metrics for next rollout
        self.metrics.clear()
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Log any remaining metrics
        self._on_rollout_end()

