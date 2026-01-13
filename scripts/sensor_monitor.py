#!/usr/bin/env python3
"""
Script to monitor sensor data in real-time while robot is in homing pose.
Places the robot in the default homing pose and continuously prints observation space data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spot_rl.envs.spot_env import CustomSpotEnv
from spot_rl.envs.observation_builder import build_observation
from spot_rl.envs.utils import quat_to_roll_pitch


def format_array(arr, precision=4):
    """Format numpy array for display."""
    return np.array2string(arr, precision=precision, suppress_small=True)


def print_observation_data(data, torso_body_id, target_lin_vel, target_ang_vel, step_count):
    """Print formatted observation data."""
    # Build observation to get all components
    obs = build_observation(data, torso_body_id, target_lin_vel, target_ang_vel)
    
    # Extract individual components for better readability
    num_joint_pos = len(data.qpos[7:])
    num_joint_vel = len(data.qvel[6:])
    num_root_vel = 6
    
    joint_positions = data.qpos[7:]
    joint_velocities = data.qvel[6:]
    root_velocities = data.qvel[0:6]
    torso_xpos = data.body(torso_body_id).xpos
    torso_z_pos = torso_xpos[2]
    torso_quat = data.body(torso_body_id).xquat
    roll, pitch = quat_to_roll_pitch(torso_quat)
    
    # Clear screen (works on most terminals)
    print("\033[2J\033[H", end="")
    
    print("=" * 80)
    print(f"SPOT ROBOT SENSOR MONITOR - Step: {step_count}")
    print("=" * 80)
    print()
    
    # Joint Positions
    print("JOINT POSITIONS (rad):")
    print(f"  {format_array(joint_positions)}")
    print(f"  Shape: {joint_positions.shape}")
    print()
    
    # Joint Velocities
    print("JOINT VELOCITIES (rad/s):")
    print(f"  {format_array(joint_velocities)}")
    print(f"  Shape: {joint_velocities.shape}")
    print()
    
    # Root Velocities
    print("ROOT VELOCITIES:")
    print(f"  Linear (x, y, z): {format_array(root_velocities[0:3])} m/s")
    print(f"  Angular (x, y, z): {format_array(root_velocities[3:6])} rad/s")
    print(f"  Full: {format_array(root_velocities)}")
    print()
    
    # Torso Position
    print("TORSO POSITION:")
    print(f"  X: {torso_xpos[0]:.4f} m")
    print(f"  Y: {torso_xpos[1]:.4f} m")
    print(f"  Z: {torso_z_pos:.4f} m (height)")
    print()
    
    # Torso Orientation
    print("TORSO ORIENTATION:")
    print(f"  Quaternion (w, x, y, z): {format_array(torso_quat)}")
    print(f"  Roll: {roll:.4f} rad ({np.degrees(roll):.2f}°)")
    print(f"  Pitch: {pitch:.4f} rad ({np.degrees(pitch):.2f}°)")
    print()
    
    # Target Velocities
    print("TARGET VELOCITIES:")
    print(f"  Linear (x, y): {format_array(target_lin_vel)} m/s")
    print(f"  Angular: {target_ang_vel:.4f} rad/s")
    print()
    
    # Full Observation Vector
    print("FULL OBSERVATION VECTOR:")
    print(f"  Shape: {obs.shape}")
    print(f"  First 10 values: {format_array(obs[:10])}")
    print(f"  Last 10 values: {format_array(obs[-10:])}")
    print()
    
    print("=" * 80)
    print("Press Ctrl+C to stop")
    print("=" * 80)


def main():
    """Main function to run sensor monitoring."""
    print("Initializing Spot environment...")
    env = CustomSpotEnv(render_mode="human")
    
    # Get default homing pose
    default_homing_pose = env.default_homing_pose
    
    print(f"Default homing pose: {format_array(default_homing_pose)}")
    print("Resetting environment to homing pose...")
    
    # Reset environment (this sets the robot to homing pose)
    obs, info = env.reset()
    
    print("Robot initialized in homing pose.")
    print("Starting real-time sensor monitoring...")
    print("Press Ctrl+C to stop\n")
    
    time.sleep(1)  # Brief pause before starting
    
    step_count = 0
    try:
        while True:
            # Apply zero action to maintain position (action is relative to homing pose)
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get current command velocities
            target_lin_vel = env.command_manager.target_lin_vel
            target_ang_vel = env.command_manager.target_ang_vel
            
            # Render the environment
            env.render()
            
            # Print observation data
            print_observation_data(
                env.data,
                env.torso_body_id,
                target_lin_vel,
                target_ang_vel,
                step_count
            )
            
            step_count += 1
            
            # Small delay to make output readable
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping sensor monitor...")
        env.close()
        print("Environment closed. Goodbye!")


if __name__ == "__main__":
    main()

