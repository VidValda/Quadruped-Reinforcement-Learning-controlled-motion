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

