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
import mujoco


def format_array(arr, precision=4):
    """Format numpy array for display."""
    return np.array2string(arr, precision=precision, suppress_small=True)


def get_foot_names(model, foot_body_ids):
    """Get foot names from body IDs."""
    foot_names = []
    for foot_id in foot_body_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, foot_id)
        foot_names.append(name if name else f"foot_{foot_id}")
    return foot_names


def print_observation_data(data, torso_body_id, target_lin_vel, target_ang_vel, step_count, 
                          reward_calculator, action, last_action):
    """Print formatted observation data with reward calculator verification."""
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
    
    # Get reward calculator data
    foot_contacts = reward_calculator._get_foot_contacts(data)
    foot_positions = reward_calculator._get_foot_positions(data)
    foot_names = get_foot_names(reward_calculator.model, reward_calculator.foot_body_ids)
    
    # Calculate reward to get all components
    reward, terminated, reward_info = reward_calculator(
        data, action, last_action, target_lin_vel, target_ang_vel, torso_body_id
    )
    
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
    print(f"  Target Height: {reward_calculator.target_height:.4f} m")
    print(f"  Height Error: {abs(torso_z_pos - reward_calculator.target_height):.4f} m")
    print()
    
    # Torso Orientation
    print("TORSO ORIENTATION:")
    print(f"  Quaternion (w, x, y, z): {format_array(torso_quat)}")
    print(f"  Roll: {roll:.4f} rad ({np.degrees(roll):.2f}°)")
    print(f"  Pitch: {pitch:.4f} rad ({np.degrees(pitch):.2f}°)")
    print()
    
    # Target Velocities
    current_lin_vel = data.body(torso_body_id).cvel[3:5]
    current_ang_vel = data.body(torso_body_id).cvel[2]
    lin_vel_error = np.linalg.norm(target_lin_vel - current_lin_vel)
    ang_vel_error = abs(target_ang_vel - current_ang_vel)
    
    print("VELOCITY TRACKING:")
    print(f"  Target Linear (x, y): {format_array(target_lin_vel)} m/s")
    print(f"  Current Linear (x, y): {format_array(current_lin_vel)} m/s")
    print(f"  Linear Error: {lin_vel_error:.4f} m/s")
    print(f"  Target Angular: {target_ang_vel:.4f} rad/s")
    print(f"  Current Angular: {current_ang_vel:.4f} rad/s")
    print(f"  Angular Error: {ang_vel_error:.4f} rad/s")
    print()
    
    # Foot Contact and Gait Information
    print("FOOT CONTACT & GAIT:")
    print(f"  Found {len(reward_calculator.foot_body_ids)} foot bodies")
    stance_feet = int(np.sum(foot_contacts))
    swing_feet = len(foot_contacts) - stance_feet
    
    for i, (foot_id, foot_name) in enumerate(zip(reward_calculator.foot_body_ids, foot_names)):
        contact_status = "STANCE" if foot_contacts[i] else "SWING"
        foot_z = foot_positions[i]
        clearance = max(0.0, foot_z - reward_calculator.min_foot_clearance)
        print(f"  {foot_name} (ID: {foot_id}): {contact_status:6s} | "
              f"Z: {foot_z:.4f} m | Clearance: {clearance:.4f} m")
    
    print(f"  Stance Feet: {stance_feet}/{len(foot_contacts)}")
    print(f"  Swing Feet: {swing_feet}/{len(foot_contacts)}")
    print(f"  Min Foot Clearance Threshold: {reward_calculator.min_foot_clearance:.4f} m")
    print()
    
    # Reward Components
    height_penalty = np.square(torso_z_pos - reward_calculator.target_height)
    height_penalty_component = -reward_calculator.height_penalty_weight * height_penalty
    
    print("REWARD CALCULATOR VERIFICATION:")
    print(f"  Total Reward: {reward:.4f}")
    print(f"  Terminated: {terminated}")
    if terminated:
        print(f"  Termination Reason: Height below threshold ({reward_calculator.termination_height_threshold:.4f} m)")
    print()
    print("  Reward Components:")
    print(f"    Linear Velocity Reward: {reward_info['rewards/lin_vel']:+.4f} "
          f"(weight: {reward_calculator.lin_vel_weight:.2f})")
    print(f"    Angular Velocity Reward: {reward_info['rewards/ang_vel']:+.4f} "
          f"(weight: {reward_calculator.ang_vel_weight:.2f})")
    print(f"    Orientation Penalty: {reward_info['rewards/orientation']:+.4f} "
          f"(weight: {reward_calculator.orientation_penalty_weight:.2f})")
    print(f"    Height Penalty: {height_penalty_component:+.4f} "
          f"(weight: {reward_calculator.height_penalty_weight:.2f}, error²: {height_penalty:.6f})")
    print(f"    Control Cost (Torques): {reward_info['rewards/torques']:+.4f} "
          f"(weight: {reward_calculator.control_cost_weight:.2f})")
    print(f"    Action Rate Penalty: {reward_info['rewards/action_rate']:+.4f} "
          f"(weight: {reward_calculator.action_rate_weight:.2f})")
    print(f"    Joint Velocity Penalty: {reward_info['rewards/joint_vel_penalty']:+.4f} "
          f"(weight: {reward_calculator.joint_vel_penalty_weight:.2f})")
    print(f"    Nominal Pose Penalty: {reward_info['rewards/nominal_pose_penalty']:+.4f} "
          f"(weight: {reward_calculator.nominal_pose_penalty_weight:.2f})")
    print(f"    Foot Clearance Reward: {reward_info['rewards/foot_clearance']:+.4f} "
          f"(weight: {reward_calculator.foot_clearance_weight:.2f})")
    print()
    
    # Additional Metrics
    print("ADDITIONAL METRICS:")
    print(f"  Action Rate: {reward_info.get('performance/action_rate', 0):.4f}")
    print(f"  Foot Clearance (avg): {reward_info.get('gait/foot_clearance', 0):.4f}")
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
    last_action = np.zeros(env.action_space.shape)
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
            
            # Print observation data with reward calculator verification
            print_observation_data(
                env.data,
                env.torso_body_id,
                target_lin_vel,
                target_ang_vel,
                step_count,
                env.reward_calculator,
                action,
                last_action
            )
            
            last_action = action
            step_count += 1
            
            # Small delay to make output readable
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping sensor monitor...")
        env.close()
        print("Environment closed. Goodbye!")


if __name__ == "__main__":
    main()

