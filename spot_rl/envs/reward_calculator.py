import numpy as np
import mujoco

from spot_rl.envs.utils import quat_to_roll_pitch, global_to_local_velocity


class SpotRewardCalculator:
    def __init__(
        self,
        target_height: float,
        model: mujoco.MjModel,
        default_homing_pose: np.ndarray,
        dt: float = 0.02,
        termination_reward: float = -20.0,
        termination_height_threshold: float = 0.26,
    ) -> None:
        self.target_height = target_height
        self.model = model
        self.default_homing_pose = default_homing_pose
        self.dt = dt
        self.termination_height_threshold = termination_height_threshold
        self.termination_reward = termination_reward
        
        # Paper weights (normalized for dt)
        # Using dt=0.02 as default (frame_skip=5 * timestep=0.004)
        self.lin_vel_weight = 1.0 * dt  # +1.0 per dt
        self.ang_vel_weight = 0.5 * dt  # +0.5 per dt
        self.lin_vel_z_penalty_weight = -2.0 * dt  # -2.0 per dt (squared)
        self.ang_vel_xy_penalty_weight = -0.05 * dt  # -0.05 per dt (squared)
        self.joint_torques_weight = -0.0002 * dt  # -0.0002 per dt
        self.action_rate_weight = -0.01 * dt  # -0.01 per dt
        self.collision_weight = -0.01 * dt  # -0.01 per dt (or -0.001 as mentioned)
        
        # Find collision geoms (knees and base)
        self.collision_geom_ids = []
        self._find_collision_geoms()
    
    def _find_collision_geoms(self):
        """Find geom IDs for knees and base that should not hit the floor."""
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        # Search for knee and base geoms
        knee_keywords = ["knee", "uleg", "upper_leg", "thigh"]
        base_keywords = ["base", "body", "torso", "chassis"]
        
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name:
                geom_name_lower = geom_name.lower()
                # Skip floor and foot geoms
                if "floor" in geom_name_lower or "foot" in geom_name_lower:
                    continue
                # Check if it's a knee or base geom
                if any(keyword in geom_name_lower for keyword in knee_keywords + base_keywords):
                    self.collision_geom_ids.append(i)
        
        # If no specific geoms found, use all non-floor, non-foot geoms
        if len(self.collision_geom_ids) == 0:
            for i in range(self.model.ngeom):
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if geom_name and "floor" not in geom_name.lower() and "foot" not in geom_name.lower():
                    self.collision_geom_ids.append(i)
    
    def _get_collisions(self, data):
        """Count collisions between knees/base and floor."""
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_geom_id == -1:
            return 0
        
        collision_count = 0
        for j in range(data.ncon):
            contact = data.contact[j]
            if contact.dist < 0.001:  # In contact
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                # Check if collision involves floor and a collision geom
                involves_floor = (geom1_id == floor_geom_id or geom2_id == floor_geom_id)
                involves_collision_geom = (geom1_id in self.collision_geom_ids or 
                                          geom2_id in self.collision_geom_ids)
                
                if involves_floor and involves_collision_geom:
                    collision_count += 1
        
        return collision_count
    
    def __call__(self, data, action, last_action, target_lin_vel, target_ang_vel, torso_body_id: int):
        # Get global frame velocities from MuJoCo
        # cvel format: [wx, wy, wz, vx, vy, vz] in global frame
        global_ang_vel_3d = data.body(torso_body_id).cvel[0:3]  # [wx, wy, wz] in global frame
        global_lin_vel_3d = data.body(torso_body_id).cvel[3:6]  # [vx, vy, vz] in global frame
        torso_z_pos = data.body(torso_body_id).xpos[2]
        torso_quat = data.body(torso_body_id).xquat

        # Transform global velocities to local (robot) frame
        local_lin_vel_3d = global_to_local_velocity(global_lin_vel_3d, torso_quat)
        local_ang_vel_3d = global_to_local_velocity(global_ang_vel_3d, torso_quat)
        
        # Extract components for tracking
        current_lin_vel_xy = local_lin_vel_3d[:2]  # [vx_local, vy_local]
        current_lin_vel_z = local_lin_vel_3d[2]  # vz_local (vertical velocity)
        current_ang_vel_z = local_ang_vel_3d[2]  # wz_local (yaw rate)
        current_ang_vel_xy = local_ang_vel_3d[:2]  # [wx_local, wy_local] (roll/pitch rates)

        # Paper reward formulation: φ(x) = exp(-x²/0.25)
        lin_vel_error = np.sum(np.square(target_lin_vel - current_lin_vel_xy))
        ang_vel_error = np.square(target_ang_vel - current_ang_vel_z)
        
        lin_vel_reward = np.exp(-lin_vel_error / 0.25)
        ang_vel_reward = np.exp(-ang_vel_error / 0.25)

        # Penalties (squared terms)
        lin_vel_z_penalty = np.square(current_lin_vel_z)  # Penalize vertical bouncing
        ang_vel_xy_penalty = np.sum(np.square(current_ang_vel_xy))  # Penalize roll/pitch rates

        # Action rate penalty: ||a_t - a_{t-1}||²
        action_rate_penalty = np.sum(np.square(action - last_action))
        
        # Joint torques penalty: ||τ||²
        # Get joint torques from MuJoCo (actuator forces)
        # Try multiple ways to access actuator forces
        if hasattr(data, 'actuator_force') and len(data.actuator_force) > 0:
            joint_torques = data.actuator_force[:]
        elif hasattr(data, 'qfrc_actuator') and len(data.qfrc_actuator) >= self.model.nu:
            # qfrc_actuator includes root forces, so skip first 6 (root) and take joint torques
            joint_torques = data.qfrc_actuator[6:6+self.model.nu] if len(data.qfrc_actuator) > 6 else data.qfrc_actuator
        else:
            # Fallback: use action as proxy (less accurate but better than zero)
            joint_torques = action
        joint_torques_penalty = np.sum(np.square(joint_torques))
        
        # Collision penalty: count collisions between knees/base and floor
        collision_count = self._get_collisions(data)

        # Compute total reward (weights already include dt normalization)
        reward = (
            self.lin_vel_weight * lin_vel_reward
            + self.ang_vel_weight * ang_vel_reward
            + self.lin_vel_z_penalty_weight * lin_vel_z_penalty
            + self.ang_vel_xy_penalty_weight * ang_vel_xy_penalty
            + self.joint_torques_weight * joint_torques_penalty
            + self.action_rate_weight * action_rate_penalty
            + self.collision_weight * collision_count
        )

        terminated = torso_z_pos < self.termination_height_threshold
        if terminated:
            reward = self.termination_reward

        # Calculate individual reward components for logging
        lin_vel_reward_component = self.lin_vel_weight * lin_vel_reward
        ang_vel_reward_component = self.ang_vel_weight * ang_vel_reward
        lin_vel_z_penalty_component = self.lin_vel_z_penalty_weight * lin_vel_z_penalty
        ang_vel_xy_penalty_component = self.ang_vel_xy_penalty_weight * ang_vel_xy_penalty
        joint_torques_component = self.joint_torques_weight * joint_torques_penalty
        action_rate_component = self.action_rate_weight * action_rate_penalty
        collision_component = self.collision_weight * collision_count
        
        roll, pitch = quat_to_roll_pitch(torso_quat)
        
        info = {
            "lin_vel_error": float(lin_vel_error),
            "ang_vel_error": float(np.sqrt(ang_vel_error)),
            "torso_height": float(torso_z_pos),
            "roll": float(roll),
            "pitch": float(pitch),
            # Reward components for TensorBoard
            "rewards/lin_vel": float(lin_vel_reward_component),
            "rewards/ang_vel": float(ang_vel_reward_component),
            "rewards/lin_vel_z_penalty": float(lin_vel_z_penalty_component),
            "rewards/ang_vel_xy_penalty": float(ang_vel_xy_penalty_component),
            "rewards/joint_torques": float(joint_torques_component),
            "rewards/action_rate": float(action_rate_component),
            "rewards/collisions": float(collision_component),
            # Tracking metrics
            "tracking/linear_velocity_error": float(lin_vel_error),
            "tracking/angular_velocity_error": float(np.sqrt(ang_vel_error)),
            "tracking/linear_velocity_z": float(current_lin_vel_z),
            "tracking/angular_velocity_xy": float(np.linalg.norm(current_ang_vel_xy)),
            # Performance metrics
            "performance/action_rate": float(np.sqrt(action_rate_penalty)),
            "performance/collision_count": int(collision_count),
        }

        return reward, terminated, info

