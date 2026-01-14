import numpy as np

from spot_rl.envs.utils import quat_to_roll_pitch, global_to_local_velocity


def build_observation(data, torso_body_id: int, target_lin_vel, target_ang_vel: float) -> np.ndarray:
    torso_xpos = data.body(torso_body_id).xpos
    torso_quat = data.body(torso_body_id).xquat
    torso_z_pos = torso_xpos[2]
    
    roll, pitch = quat_to_roll_pitch(torso_quat)
    pitch_roll = np.array([pitch, roll])

    global_lin_vel = data.qvel[0:3]  # [vx, vy, vz] in global frame
    global_ang_vel = data.qvel[3:6]  # [wx, wy, wz] in global frame
    
    local_lin_vel = global_to_local_velocity(global_lin_vel, torso_quat)
    local_ang_vel = global_to_local_velocity(global_ang_vel, torso_quat)
    
    local_root_vel = np.concatenate([local_lin_vel, local_ang_vel])

    return np.concatenate(
        [
            data.qpos[7:],
            data.qvel[6:],
            local_root_vel,
            np.array([torso_z_pos]),
            pitch_roll,
            target_lin_vel,
            np.array([target_ang_vel]),
        ]
    ).astype(np.float32)

