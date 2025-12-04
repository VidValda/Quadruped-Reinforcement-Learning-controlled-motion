import numpy as np


def build_observation(data, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    torso_xpos = data.body(torso_body_id).xpos
    torso_quat = data.body(torso_body_id).xquat
    torso_z_pos = torso_xpos[2]
    
    # Convert quaternion [w, x, y, z] to Euler angles and extract only pitch and roll
    w, x, y, z = torso_quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(2 * (w * y - z * x))
    pitch_roll = np.array([pitch, roll])

    return np.concatenate(
        [
            data.qpos[7:],
            data.qvel[6:],
            data.qvel[0:6],
            np.array([torso_z_pos]),
            pitch_roll,
            target_lin_vel,
            np.array([target_ang_vel]),
        ]
    ).astype(np.float32)

