from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


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
        # Counter to track if we've found any metrics (for debugging)
        self.found_metrics_count = 0
    
    def _on_step(self) -> bool:
        """Called at each step. Extract metrics from info dict."""
        # Access infos from locals - in vectorized envs, infos is a list of dicts
        # This is the most reliable way to get infos during rollout
        infos = None
        
        # Try multiple ways to access infos from locals
        try:
            if hasattr(self.locals, 'infos'):
                infos = self.locals.infos
            elif isinstance(self.locals, dict):
                infos = self.locals.get('infos', None)
            elif hasattr(self.locals, '__dict__'):
                infos = getattr(self.locals, 'infos', None)
        except Exception:
            pass
        
        # Extract metrics if infos are available
        if infos is not None:
            self._extract_metrics_from_infos(infos)
        
        return True
    
    def _extract_metrics_from_infos(self, infos: Any) -> None:
        """Extract metrics from infos (can be list of dicts or single dict)."""
        # List of metric keys to extract and log
        metric_keys = [
            # Reward components
            "rewards/lin_vel",
            "rewards/ang_vel",
            "rewards/orientation",
            "rewards/torques",
            "rewards/action_rate",
            "rewards/joint_vel_penalty",
            "rewards/nominal_pose_penalty",
            "rewards/foot_clearance",
            # Tracking metrics
            "tracking/linear_velocity_error",
            "tracking/angular_velocity_error",
            # Performance metrics
            "performance/action_rate",
            # Gait metrics
            "gait/stance_feet",
            "gait/swing_feet",
            "gait/foot_clearance",
        ]
        
        # Handle different info formats
        if isinstance(infos, (list, tuple)):
            # Vectorized environment: list/tuple of info dicts (one per environment)
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
        # Debug: Confirm callback is being called
        if not hasattr(self, '_debug_printed'):
            print("[TensorBoardMetricsCallback] Callback is active and being called")
            self._debug_printed = True
        
        # Try multiple methods to get infos
        
        # Method 1: Try to get from rollout buffer
        rollout_infos = None
        if hasattr(self.model, 'rollout_buffer'):
            if hasattr(self.model.rollout_buffer, 'infos'):
                rollout_infos = self.model.rollout_buffer.infos
        
        # Method 2: Try to get from locals
        if rollout_infos is None:
            if hasattr(self.locals, 'infos'):
                rollout_infos = self.locals.infos
            elif isinstance(self.locals, dict) and 'infos' in self.locals:
                rollout_infos = self.locals['infos']
        
        # Process rollout_infos if available
        if rollout_infos is not None:
            # Rollout buffer infos can be:
            # - A list of lists: [step][env] 
            # - A list/tuple of dicts
            # - A tuple of tuples: ((dict, dict, ...), (dict, dict, ...), ...)
            # - A single dict
            all_infos = []
            
            # Debug: Print structure of rollout_infos
            if not hasattr(self, '_debug_structure_printed'):
                print(f"[TensorBoardMetricsCallback] DEBUG: rollout_infos type: {type(rollout_infos)}")
                if isinstance(rollout_infos, (list, tuple)) and len(rollout_infos) > 0:
                    print(f"[TensorBoardMetricsCallback] DEBUG: First element type: {type(rollout_infos[0])}")
                    if isinstance(rollout_infos[0], (list, tuple)) and len(rollout_infos[0]) > 0:
                        print(f"[TensorBoardMetricsCallback] DEBUG: First nested element type: {type(rollout_infos[0][0])}")
                        if isinstance(rollout_infos[0][0], dict) and len(rollout_infos[0][0]) > 0:
                            print(f"[TensorBoardMetricsCallback] DEBUG: First dict keys: {list(rollout_infos[0][0].keys())[:5]}")
                self._debug_structure_printed = True
            
            if isinstance(rollout_infos, (list, tuple)):
                for step_infos in rollout_infos:
                    if isinstance(step_infos, (list, tuple)):
                        # Multiple environments: list/tuple of info dicts
                        all_infos.extend(step_infos)
                    elif isinstance(step_infos, dict):
                        # Single environment or single info dict
                        all_infos.append(step_infos)
            elif isinstance(rollout_infos, dict):
                all_infos.append(rollout_infos)
            
            # Extract metrics from all collected infos
            for info in all_infos:
                if isinstance(info, dict):
                    self._extract_metrics_from_infos(info)
                    self.found_metrics_count += 1
                elif isinstance(info, (list, tuple)):
                    # If info is itself a list/tuple, try to extract from it
                    for item in info:
                        if isinstance(item, dict):
                            self._extract_metrics_from_infos(item)
                            self.found_metrics_count += 1
        
        # Log accumulated metrics from the rollout
        if self.metrics and self.logger is not None:
            for metric_name, values in self.metrics.items():
                if values:
                    mean_value = np.mean(values)
                    self.logger.record(metric_name, mean_value)
        
        # Debug: Print information about metrics found
        if self.metrics:
            print(f"[TensorBoardMetricsCallback] Logging {len(self.metrics)} metric types: {list(self.metrics.keys())}")
            for metric_name, values in self.metrics.items():
                if values:
                    mean_value = np.mean(values)
                    print(f"  - {metric_name}: {mean_value:.4f} (from {len(values)} samples)")
        else:
            # Debug: Print why we didn't find metrics
            if rollout_infos is None:
                print(f"[TensorBoardMetricsCallback] WARNING: No infos found in rollout buffer or locals")
            else:
                print(f"[TensorBoardMetricsCallback] WARNING: Found infos but no metrics extracted. Infos type: {type(rollout_infos)}")
        
        # Reset metrics for next rollout
        self.metrics.clear()
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Log any remaining metrics
        self._on_rollout_end()

