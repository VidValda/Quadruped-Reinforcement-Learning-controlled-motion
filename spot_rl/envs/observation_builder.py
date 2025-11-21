import numpy as np


def build_observation(data, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    torso_xpos = data.body(torso_body_id).xpos
    torso_quat = data.body(torso_body_id).xquat
    torso_z_pos = torso_xpos[2]

    return np.concatenate(
        [
            data.qpos[7:],
            data.qvel[6:],
            data.qvel[0:6],
            np.array([torso_z_pos]),
            torso_quat,
            target_lin_vel,
            np.array([target_ang_vel]),
        ]
    ).astype(np.float32)

