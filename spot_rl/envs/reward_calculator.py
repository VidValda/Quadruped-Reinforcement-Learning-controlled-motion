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

    def __call__(self, data, action, last_action, target_lin_vel, target_ang_vel, torso_body_id: int):
        current_lin_vel = data.body(torso_body_id).cvel[3:5]
        current_ang_vel = data.body(torso_body_id).cvel[2]
        torso_z_pos = data.body(torso_body_id).xpos[2]
        torso_quat = data.body(torso_body_id).xquat

        lin_vel_error = np.linalg.norm(target_lin_vel - current_lin_vel)
        ang_vel_error = np.square(target_ang_vel - current_ang_vel)

        lin_vel_reward = np.exp(-1.5 * lin_vel_error)
        ang_vel_reward = np.exp(-1.0 * ang_vel_error)

        roll, pitch = self._quat_to_roll_pitch(torso_quat)

        height_penalty = np.square(torso_z_pos - self.target_height)
        orientation_penalty = np.square(roll) + np.square(pitch)

        action_rate_penalty = np.sum(np.square(action - last_action))
        control_cost = np.sum(np.square(action))

        reward = (
            self.lin_vel_weight * lin_vel_reward
            + self.ang_vel_weight * ang_vel_reward
            - self.height_penalty_weight * height_penalty
            - self.orientation_penalty_weight * orientation_penalty
            - self.action_rate_weight * action_rate_penalty
            - self.control_cost_weight * control_cost
        )

        terminated = torso_z_pos < self.termination_height_threshold
        if terminated:
            reward = self.termination_reward

        info = {
            "lin_vel_error": float(lin_vel_error),
            "ang_vel_error": float(ang_vel_error),
            "torso_height": float(torso_z_pos),
            "roll": float(roll),
            "pitch": float(pitch),
        }

        return reward, terminated, info

