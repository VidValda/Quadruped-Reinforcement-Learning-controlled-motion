import numpy as np


def quat_to_roll_pitch(quat):

    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Calculate roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Calculate pitch with safety check for arcsin domain
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    return roll, pitch


def quat_to_rot_matrix(quat):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    The quaternion represents rotation from local to global frame.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def global_to_local_velocity(global_vel, quat):
    """
    Transform velocity vector from global (world) frame to local (robot) frame.
    
    Args:
        global_vel: 3D velocity vector in global frame [vx, vy, vz]
        quat: Quaternion [w, x, y, z] representing robot orientation (local to global)
    
    Returns:
        3D velocity vector in local frame [vx_local, vy_local, vz_local]
    """
    R = quat_to_rot_matrix(quat)
    # R transforms from local to global, so R^T transforms from global to local
    local_vel = R.T @ global_vel
    return local_vel

