import numpy as np


class SpotRewardCalculator:
    def __init__(
        self,
        target_height: float,
        lin_vel_weight: float = 2.0,
        ang_vel_weight: float = 1.0,
        height_penalty_weight: float = 2.0,
        orientation_penalty_weight: float = 1.0,
        action_rate_weight: float = 1,
        control_cost_weight: float = 0.03,
        termination_height_threshold: float = 0.2,
        termination_reward: float = -10.0,
    ) -> None:
        self.target_height = target_height
        self.lin_vel_weight = lin_vel_weight
        self.ang_vel_weight = ang_vel_weight
        self.height_penalty_weight = height_penalty_weight
        self.orientation_penalty_weight = orientation_penalty_weight
        self.action_rate_weight = action_rate_weight
        self.control_cost_weight = control_cost_weight
        self.termination_height_threshold = termination_height_threshold
        self.termination_reward = termination_reward

    @staticmethod
    def _quat_to_roll_pitch(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        return roll, pitch

    def __call__(self, data, action, last_action, target_lin_vel, target_ang_vel, torso_body_id: int, manual_control: bool = False):
        current_lin_vel = data.body(torso_body_id).cvel[3:5]
        current_ang_vel = data.body(torso_body_id).cvel[2]
        torso_z_pos = data.body(torso_body_id).xpos[2]
        torso_quat = data.body(torso_body_id).xquat

        reward = 0.0

        # 1. Reward for following target velocities (only in automatic mode)
        if not manual_control:
            lin_vel_error = np.linalg.norm(current_lin_vel - target_lin_vel)
            ang_vel_error = abs(current_ang_vel - target_ang_vel)
            
            reward += 3.0 * np.exp(-2.0 * lin_vel_error)
            reward += 2.0 * np.exp(-1.5 * ang_vel_error)

        # 2. Penalty for incorrect height
        height_error = abs(torso_z_pos - self.target_height)
        reward -= 1.0 * height_error

        # 3. Penalty for tilt
        roll, pitch = self._quat_to_roll_pitch(torso_quat)
        orientation_penalty = roll**2 + pitch**2
        reward -= 0.5 * orientation_penalty

        # 4. Termination for falling
        terminated = torso_z_pos < self.termination_height_threshold
        if terminated:
            reward -= 10.0

        # 5. Penalty for abrupt actions
        action_penalty = np.sum(np.square(action - last_action))
        reward -= 0.01 * action_penalty

        info = {
            "lin_vel_error": float(np.linalg.norm(current_lin_vel - target_lin_vel)) if not manual_control else 0.0,
            "ang_vel_error": float(abs(current_ang_vel - target_ang_vel)) if not manual_control else 0.0,
            "torso_height": float(torso_z_pos),
            "roll": float(roll),
            "pitch": float(pitch),
        }

        return reward, terminated, info

