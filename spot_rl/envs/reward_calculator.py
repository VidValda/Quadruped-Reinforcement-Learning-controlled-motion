import numpy as np
import mujoco

from spot_rl.envs.utils import quat_to_roll_pitch


class SpotRewardCalculator:
    def __init__(
        self,
        target_height: float,
        model: mujoco.MjModel,
        default_homing_pose: np.ndarray,
        lin_vel_weight: float = 2.0,
        ang_vel_weight: float = 1.0,
        height_penalty_weight: float = 3.0,
        orientation_penalty_weight: float = 1.0,
        action_rate_weight: float = 0.1,  
        control_cost_weight: float = 0.05,
        joint_vel_penalty_weight: float = 0.003,
        nominal_pose_penalty_weight: float = 0.5,
        foot_clearance_weight: float = 1.0,
        termination_height_threshold: float = 0.23,
        termination_reward: float = -10.0,
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
        
        # Find foot body IDs (common Spot naming conventions)
        self.foot_body_ids = []
        
        # DEBUG: Print all body names to see what's available
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
        
        # DEBUG: Print all geom names to see what's available
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
        
        # Try to find foot bodies by exact name match (common Spot naming conventions)
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
        
        # If no foot bodies found by name, try to find them by searching for bodies with "foot" in name
        if len(self.foot_body_ids) == 0:
            print("\nDEBUG: No exact matches found. Searching for bodies with 'foot' in name:")
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and "foot" in body_name.lower():
                    self.foot_body_ids.append(i)
                    print(f"  ✓ Found '{body_name}' (ID {i}) - contains 'foot'")
        
        # If still no feet found, use lower leg bodies (lleg = lower leg, which typically represents the foot)
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
        
        # Print final result
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
        """Detect which feet are in contact with the ground (Stance Phase)."""
        foot_contacts = np.zeros(len(self.foot_body_ids), dtype=bool)
        
        # Find floor geom ID (usually named "floor")
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        # Check contacts for each foot
        for i, foot_id in enumerate(self.foot_body_ids):
            # Check if any contact involves this foot body
            for j in range(data.ncon):
                contact = data.contact[j]
                # Check if contact involves the foot body
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                # Get body IDs for the geoms
                body1_id = self.model.geom_bodyid[geom1_id]
                body2_id = self.model.geom_bodyid[geom2_id]
                
                # Check if contact is with floor and involves foot
                # dist < 0 means penetration (contact), dist > 0 means separation
                is_contact = contact.dist < 0.001  # Small threshold for contact
                involves_foot = (body1_id == foot_id or body2_id == foot_id)
                involves_floor = (floor_geom_id != -1 and (geom1_id == floor_geom_id or geom2_id == floor_geom_id))
                
                # If we have foot bodies, check if contact involves foot
                # Otherwise, check if contact involves floor (generic detection)
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
        current_lin_vel = data.body(torso_body_id).cvel[3:5]
        current_ang_vel = data.body(torso_body_id).cvel[2]
        torso_z_pos = data.body(torso_body_id).xpos[2]
        torso_quat = data.body(torso_body_id).xquat

        lin_vel_error = np.linalg.norm(target_lin_vel - current_lin_vel)
        ang_vel_error = np.square(target_ang_vel - current_ang_vel)

        lin_vel_reward = np.exp(-1.5 * lin_vel_error)
        ang_vel_reward = np.exp(-1.0 * ang_vel_error)

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
        
        # Foot Clearance Reward: Reward lifting feet during swing phase
        foot_clearance_reward = 0.0
        num_swing_feet = 0
        for i in range(len(foot_contacts)):
            if not foot_contacts[i]:  # Swing phase (foot in air)
                num_swing_feet += 1
                # Reward foot clearance above minimum threshold
                clearance = max(0.0, foot_positions[i] - self.min_foot_clearance)
                foot_clearance_reward += clearance
        
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

