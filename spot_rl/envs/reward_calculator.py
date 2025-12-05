import numpy as np

from spot_rl.envs.utils import quat_to_roll_pitch


class SpotRewardCalculator:
    def __init__(
        self,
        target_height: float,
        reward_scales: dict,
        dt: float,
        jump_reward_steps: int = 30,
        tracking_sigma: float = 0.25,
    ) -> None:
        self.target_height = target_height
        self.reward_scales = {k: v * dt for k, v in reward_scales.items()}
        self.jump_reward_steps = jump_reward_steps
        self.tracking_sigma = tracking_sigma
        
        self.episode_sums = {name: 0.0 for name in self.reward_scales.keys()}

    def reset_episode_sums(self):
        self.episode_sums = {name: 0.0 for name in self.reward_scales.keys()}

    def __call__(
        self,
        data,
        action,
        last_action,
        commands,
        base_lin_vel,
        base_ang_vel,
        base_pos,
        base_quat,
        dof_pos,
        dof_vel,
        default_dof_pos,
        jump_toggled_buf,
        jump_target_height,
        torso_body_id: int,
    ):
        reward = 0.0
        reward_dict = {}

        for name, scale in self.reward_scales.items():
            rew_func = getattr(self, f"_reward_{name}")
            rew = rew_func(
                commands,
                base_lin_vel,
                base_ang_vel,
                base_pos,
                base_quat,
                dof_pos,
                dof_vel,
                default_dof_pos,
                action,
                last_action,
                jump_toggled_buf,
                jump_target_height,
            )
            scaled_rew = rew * scale
            reward += scaled_rew
            reward_dict[name] = float(rew)
            self.episode_sums[name] += scaled_rew

        roll, pitch = quat_to_roll_pitch(base_quat)
        
        terminated = base_pos[2] < 0.2

        info = {
            "reward": float(reward),
            "roll": float(roll),
            "pitch": float(pitch),
            "torso_height": float(base_pos[2]),
            **reward_dict,
        }

        return reward, terminated, info

    def _reward_tracking_lin_vel(self, commands, base_lin_vel, *args, **kwargs):
        lin_vel_error = np.sum(np.square(commands[:2] - base_lin_vel[:2]))
        return np.exp(-lin_vel_error / self.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands, base_lin_vel, base_ang_vel, *args, **kwargs):
        ang_vel_error = np.square(commands[2] - base_ang_vel[2] if len(base_ang_vel) > 2 else base_ang_vel[0])
        return np.exp(-ang_vel_error / self.tracking_sigma)

    def _reward_lin_vel_z(self, *args, base_lin_vel, jump_toggled_buf, **kwargs):
        active_mask = 1.0 if jump_toggled_buf < 0.01 else 0.0
        return active_mask * np.square(base_lin_vel[2])

    def _reward_action_rate(self, *args, action, last_action, jump_toggled_buf, **kwargs):
        active_mask = 1.0 if jump_toggled_buf < 0.01 else 0.0
        return active_mask * np.sum(np.square(last_action - action))

    def _reward_similar_to_default(self, *args, dof_pos, default_dof_pos, jump_toggled_buf, **kwargs):
        active_mask = 1.0 if jump_toggled_buf < 0.01 else 0.0
        return active_mask * np.sum(np.abs(dof_pos - default_dof_pos))

    def _reward_base_height(self, commands, *args, base_pos, jump_toggled_buf, **kwargs):
        active_mask = 1.0 if jump_toggled_buf < 0.01 else 0.0
        return active_mask * np.square(base_pos[2] - commands[3])

    def _reward_jump_height_tracking(self, *args, base_pos, jump_toggled_buf, jump_target_height, **kwargs):
        mask = (jump_toggled_buf >= 0.3 * self.jump_reward_steps) & (jump_toggled_buf < 0.6 * self.jump_reward_steps)
        if not mask:
            return 0.0
        height_diff = np.exp(-np.square(base_pos[2] - jump_target_height))
        return height_diff

    def _reward_jump_height_achievement(self, *args, base_pos, jump_toggled_buf, jump_target_height, **kwargs):
        mask = (jump_toggled_buf >= 0.3 * self.jump_reward_steps) & (jump_toggled_buf < 0.6 * self.jump_reward_steps)
        if not mask:
            return 0.0
        binary_bonus = 1.0 if np.abs(base_pos[2] - jump_target_height) < 0.2 else 0.0
        return binary_bonus

    def _reward_jump_speed(self, *args, base_lin_vel, jump_toggled_buf, **kwargs):
        mask = (jump_toggled_buf >= 0.3 * self.jump_reward_steps) & (jump_toggled_buf < 0.6 * self.jump_reward_steps)
        if not mask:
            return 0.0
        return np.exp(base_lin_vel[2]) * 0.2

    def _reward_jump_landing(self, *args, base_pos, jump_toggled_buf, **kwargs):
        mask = jump_toggled_buf >= 0.6 * self.jump_reward_steps
        if not mask:
            return 0.0
        height_error = -np.square(base_pos[2] - self.target_height)
        return height_error
