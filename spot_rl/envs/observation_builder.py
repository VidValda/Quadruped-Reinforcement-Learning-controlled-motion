import numpy as np

from spot_rl.envs.utils import transform_by_quat, inv_quat


def build_observation(
    data,
    torso_body_id: int,
    commands: np.ndarray,
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    default_dof_pos: np.ndarray,
    actions: np.ndarray,
    jump_toggled_buf: float,
    jump_reward_steps: int,
    obs_scales: dict,
    commands_scale: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            base_ang_vel * obs_scales["ang_vel"],
            projected_gravity,
            commands * commands_scale,
            (dof_pos - default_dof_pos) * obs_scales["dof_pos"],
            dof_vel * obs_scales["dof_vel"],
            actions,
            np.array([jump_toggled_buf / jump_reward_steps]),
        ]
    ).astype(np.float32)
