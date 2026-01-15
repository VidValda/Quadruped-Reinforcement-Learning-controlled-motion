import numpy as np
import mujoco

from spot_rl.envs.utils import quat_to_roll_pitch, global_to_local_velocity


class SpotRewardCalculator:
    def __init__(
        self,
        target_height: float,
        model: mujoco.MjModel,
        default_homing_pose: np.ndarray,
        lin_vel_weight: float = 1.5,
        ang_vel_weight: float = 0.5,
        height_penalty_weight: float = 3.0,
        orientation_penalty_weight: float = 1.0,
        termination_reward: float = -20.0,
        termination_height_threshold: float = 0.26,
        action_rate_weight: float = 0.015,
        control_cost_weight: float = 0.01,
        joint_vel_penalty_weight: float = 0.006,
        nominal_pose_penalty_weight: float = 0.25,
        foot_clearance_weight: float = 0.0,
        contact_force_threshold: float = 10.0,
        min_foot_clearance: float = 0.05,
    ) -> None:
        self.target_height = target_height
        self.model = model
        self.default_homing_pose = default_homing_pose
        self.lin_vel_weight = lin_vel_weight
        self.ang_vel_weight = ang_vel_weight
        self.height_penalty_weight = height_penalty_weight
        self.orientation_penalty_weight = orientation_penalty_weight
        self.action_rate_weight = action_rate_weight
        self.control_cost_weight = control_cost_weight
        self.joint_vel_penalty_weight = joint_vel_penalty_weight
        self.nominal_pose_penalty_weight = nominal_pose_penalty_weight
        self.foot_clearance_weight = foot_clearance_weight
        self.termination_height_threshold = termination_height_threshold
        self.termination_reward = termination_reward
        self.contact_force_threshold = contact_force_threshold
        self.min_foot_clearance = min_foot_clearance
        
        self.foot_body_offsets = None
        
        self.foot_body_ids = []
        
        print("=" * 80)
        print("DEBUG: All body names in model:")
        print("=" * 80)
        all_body_names = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                all_body_names.append((i, body_name))
                print(f"  Body ID {i:3d}: '{body_name}'")
        print(f"Total bodies: {self.model.nbody}")
        print("=" * 80)
        
        print("\nDEBUG: All geom names in model:")
        print("=" * 80)
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            geom_body_id = self.model.geom_bodyid[i]
            geom_body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, geom_body_id)
            if geom_name:
                print(f"  Geom ID {i:3d}: '{geom_name}' -> Body '{geom_body_name}' (ID {geom_body_id})")
            else:
                print(f"  Geom ID {i:3d}: <unnamed> -> Body '{geom_body_name}' (ID {geom_body_id})")
        print(f"Total geoms: {self.model.ngeom}")
        print("=" * 80)
        
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot",
                      "fl_foot", "fr_foot", "rl_foot", "rr_foot",
                      "foot_fl", "foot_fr", "foot_rl", "foot_rr",
                      "FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link",
                      "fl_foot_link", "fr_foot_link", "rl_foot_link", "rr_foot_link"]
        
        print("\nDEBUG: Searching for foot bodies by name:")
        print(f"  Searching for: {foot_names}")
        for foot_name in foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
            if foot_id != -1:
                self.foot_body_ids.append(foot_id)
                print(f"  ✓ Found '{foot_name}' -> ID {foot_id}")
            else:
                print(f"  ✗ Not found: '{foot_name}'")
        
        if len(self.foot_body_ids) == 0:
            print("\nDEBUG: No exact matches found. Searching for bodies with 'foot' in name:")
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and "foot" in body_name.lower():
                    self.foot_body_ids.append(i)
                    print(f"  ✓ Found '{body_name}' (ID {i}) - contains 'foot'")
        
        if len(self.foot_body_ids) == 0:
            print("\nDEBUG: No foot bodies found. Using lower leg bodies as feet:")
            lower_leg_names = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
            for leg_name in lower_leg_names:
                leg_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, leg_name)
                if leg_id != -1:
                    self.foot_body_ids.append(leg_id)
                    print(f"  ✓ Using '{leg_name}' (ID {leg_id}) as foot body")
                else:
                    print(f"  ✗ Not found: '{leg_name}'")
        
        print("\n" + "=" * 80)
        print(f"FINAL: Found {len(self.foot_body_ids)} foot bodies: {self.foot_body_ids}")
        if len(self.foot_body_ids) > 0:
            print("Foot body names:")
            for foot_id in self.foot_body_ids:
                foot_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_id)
                print(f"  ID {foot_id}: '{foot_name}'")
        else:
            print("WARNING: No foot bodies found! Foot contact detection will not work.")
        print("=" * 80 + "\n")

    def _get_foot_contacts(self, data):
        foot_contacts = np.zeros(len(self.foot_body_ids), dtype=bool)
        
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        for i, foot_id in enumerate(self.foot_body_ids):
            for j in range(data.ncon):
                contact = data.contact[j]
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                body1_id = self.model.geom_bodyid[geom1_id]
                body2_id = self.model.geom_bodyid[geom2_id]
                
                is_contact = contact.dist < 0.001
                involves_foot = (body1_id == foot_id or body2_id == foot_id)
                involves_floor = (floor_geom_id != -1 and (geom1_id == floor_geom_id or geom2_id == floor_geom_id))
                
                if involves_foot and (involves_floor or floor_geom_id == -1) and is_contact:
                    foot_contacts[i] = True
                    break
        
        return foot_contacts
    
    def _get_foot_positions(self, data):
        """Get foot positions (z-coordinate for clearance calculation)."""
        foot_positions = np.zeros(len(self.foot_body_ids))
        for i, foot_id in enumerate(self.foot_body_ids):
            foot_positions[i] = data.body(foot_id).xpos[2]
        return foot_positions
    
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
        # Extract only x and y components for 2D movement tracking
        current_lin_vel = local_lin_vel_3d[:2]  # [vx_local, vy_local]
        current_ang_vel = local_ang_vel_3d[2]  # wz_local (yaw rate in local frame)


        lin_vel_error = np.sum(np.square(target_lin_vel - current_lin_vel))
        ang_vel_error = np.square(target_ang_vel - current_ang_vel)

        lin_vel_reward = np.exp(-lin_vel_error / 0.25)
        ang_vel_reward = np.exp(-ang_vel_error / 0.25)

        roll, pitch = quat_to_roll_pitch(torso_quat)

        height_penalty = np.square(torso_z_pos - self.target_height)
        orientation_penalty = np.square(roll) + np.square(pitch)

        action_rate_penalty = np.sum(np.square(action - last_action))
        control_cost = np.sum(np.square(action))
        
        # Joint Velocity Penalty: Penalize frantic joint movements
        joint_velocities = data.qvel[6:]  # Skip root velocities (first 6)
        joint_vel_penalty = np.sum(np.square(joint_velocities))
        
        # Nominal Pose Penalty: Penalize deviation from standing pose
        current_joint_positions = data.qpos[7:]  # Skip root position (first 7: x, y, z, quat)
        if len(current_joint_positions) == len(self.default_homing_pose):
            joint_pos_error = current_joint_positions - self.default_homing_pose
            nominal_pose_penalty = np.sum(np.square(joint_pos_error))
        else:
            nominal_pose_penalty = 0.0
        
        # Foot Contact Detection (Stance vs Swing Phase)
        foot_contacts = self._get_foot_contacts(data)
        foot_positions = self._get_foot_positions(data)
        
        # Estimate foot body offset from stance feet (where foot tip is at ground level)
        if self.foot_body_offsets is None:
            self.foot_body_offsets = np.full(len(self.foot_body_ids), 0.26)
        for i in range(len(foot_contacts)):
            if foot_contacts[i]:
                self.foot_body_offsets[i] = foot_positions[i]
        
        # Foot Clearance Reward: Reward lifting feet during swing phase
        foot_clearance_reward = 0.0
        num_swing_feet = 0
        target_clearance = 0.07  # The "Perfect" step height (7cm)

        for i in range(len(foot_contacts)):
            if not foot_contacts[i]:  # Swing phase (foot in air)
                num_swing_feet += 1
                
                # Calculate foot tip height relative to ground
                if self.foot_body_offsets[i] > 0:
                    foot_tip_z = foot_positions[i] - self.foot_body_offsets[i]
                else:
                    foot_tip_z = foot_positions[i] * 0.5
                
                # BELL CURVE LOGIC:
                # Reward peaks at target_clearance, decays if too low OR too high.
                # The '150' controls the strictness (higher = narrower curve).
                foot_clearance_reward += np.exp(-150 * (foot_tip_z - target_clearance)**2)
        
        # Normalize by number of swing feet (avoid division by zero)
        if num_swing_feet > 0:
            foot_clearance_reward = foot_clearance_reward / num_swing_feet

        reward = (
            self.lin_vel_weight * lin_vel_reward
            + self.ang_vel_weight * ang_vel_reward
            - self.height_penalty_weight * height_penalty
            - self.orientation_penalty_weight * orientation_penalty
            - self.action_rate_weight * action_rate_penalty
            - self.control_cost_weight * control_cost
            - self.joint_vel_penalty_weight * joint_vel_penalty
            - self.nominal_pose_penalty_weight * nominal_pose_penalty
            + self.foot_clearance_weight * foot_clearance_reward
        )

        terminated = torso_z_pos < self.termination_height_threshold
        if terminated:
            reward = self.termination_reward

        # Calculate individual reward components for logging
        lin_vel_reward_component = self.lin_vel_weight * lin_vel_reward
        ang_vel_reward_component = self.ang_vel_weight * ang_vel_reward
        orientation_penalty_component = -self.orientation_penalty_weight * orientation_penalty
        control_cost_component = -self.control_cost_weight * control_cost
        action_rate_component = -self.action_rate_weight * action_rate_penalty
        joint_vel_penalty_component = -self.joint_vel_penalty_weight * joint_vel_penalty
        nominal_pose_penalty_component = -self.nominal_pose_penalty_weight * nominal_pose_penalty
        foot_clearance_reward_component = self.foot_clearance_weight * foot_clearance_reward
        height_penalty_component = -self.height_penalty_weight * height_penalty
        info = {
            "lin_vel_error": float(lin_vel_error),
            "ang_vel_error": float(np.sqrt(ang_vel_error)),  # Convert squared error to absolute error
            "torso_height": float(torso_z_pos),
            "roll": float(roll),
            "pitch": float(pitch),
            # Reward components for TensorBoard
            "rewards/lin_vel": float(lin_vel_reward_component),
            "rewards/ang_vel": float(ang_vel_reward_component),
            "rewards/orientation": float(orientation_penalty_component),
            "rewards/torques": float(control_cost_component),
            "rewards/action_rate": float(action_rate_component),
            "rewards/joint_vel_penalty": float(joint_vel_penalty_component),
            "rewards/nominal_pose_penalty": float(nominal_pose_penalty_component),
            "rewards/foot_clearance": float(foot_clearance_reward_component),
            "rewards/height_penalty": float(height_penalty_component),
            # Tracking metrics
            "tracking/linear_velocity_error": float(lin_vel_error),
            "tracking/angular_velocity_error": float(np.sqrt(ang_vel_error)),
            # Performance metrics
            "performance/action_rate": float(np.sqrt(action_rate_penalty)),  # Use sqrt for better scale
            # Stance/Swing phase tracking
            "gait/stance_feet": int(np.sum(foot_contacts)),
            "gait/swing_feet": int(num_swing_feet),
            "gait/foot_clearance": float(foot_clearance_reward),
        }

        return reward, terminated, info

