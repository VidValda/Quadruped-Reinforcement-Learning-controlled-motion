import numpy as np


def quat_to_roll_pitch(quat):
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


def quat_to_xyz(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def inv_quat(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def transform_by_quat(vec, quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    vec_quat = np.array([0.0, vec[0], vec[1], vec[2]])
    
    quat_conj = inv_quat(quat)
    
    result_quat = quat_mult(quat_mult(quat, vec_quat), quat_conj)
    
    return result_quat[1:4]


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def transform_quat_by_quat(quat1, quat2):
    return quat_mult(quat2, quat1)

