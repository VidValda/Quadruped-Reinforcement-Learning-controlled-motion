import numpy as np


def build_observation(data, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    torso_xpos = data.body(torso_body_id).xpos
    torso_quat = data.body(torso_body_id).xquat
    
    w, x, y, z = torso_quat[0], torso_quat[1], torso_quat[2], torso_quat[3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(2 * (w * y - z * x))
    
    current_lin_vel = data.body(torso_body_id).cvel[3:6]
    current_ang_vel = data.body(torso_body_id).cvel[0:3]

    return np.concatenate(
        [
            data.qpos[7:],  # Joint positions
            data.qvel[6:],  # Joint velocities
            current_lin_vel,  # Linear velocity (from cvel)
            current_ang_vel,  # Angular velocity (from cvel)
            np.array([roll, pitch]),  # Roll and pitch instead of full quaternion
            np.array([torso_xpos[2]]),  # Height
            target_lin_vel,  # Target linear velocity
            np.array([target_ang_vel]),  # Target angular velocity
        ]
    ).astype(np.float32)

