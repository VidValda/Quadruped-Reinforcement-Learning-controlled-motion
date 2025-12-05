import numpy as np

from spot_rl.envs.utils import quat_to_roll_pitch


def build_observation(data, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    torso_xpos = data.body(torso_body_id).xpos
    torso_quat = data.body(torso_body_id).xquat
    torso_z_pos = torso_xpos[2]
    
    roll, pitch = quat_to_roll_pitch(torso_quat)
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

