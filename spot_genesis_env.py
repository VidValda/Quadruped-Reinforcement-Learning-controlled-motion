import torch
import math
import genesis as gs
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# --- Helper Functions (Ported from Go2) ---
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def gs_rand_gaussian(mean, min, max, n_std, shape, device):
    mean_tensor = mean.expand(shape).to(device)
    std_tensor = torch.full(shape, (max - min)/ 4.0 * n_std, device=device)
    return torch.clamp(torch.normal(mean_tensor, std_tensor), min, max)

class SpotGenesisEnv:
    """
    Genesis-based environment for Boston Dynamics Spot, adapted to match
    the feature set of the Go2 environment (jumping, shaped rewards).
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # --- Init Base Pos/Quat (MOVED UP) ---
        # Must be defined before creating the entity in the scene
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # --- Scene Setup ---
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            # Updated to fix deprecation warning: n_rendered_envs -> rendered_envs_idx
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=True),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # --- Robot Setup (Spot) ---
        robot_path = self.env_cfg.get("robot_path", "urdf/spot/spot.urdf")
        print(f"Loading Spot from: {robot_path}")

        if robot_path.endswith(".mjcf") or robot_path.endswith(".xml"):
             self.robot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=robot_path,
                    pos=self.base_init_pos.cpu().numpy(),
                    quat=self.base_init_quat.cpu().numpy(),
                ),
            )
        else:
             self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=robot_path,
                    pos=self.base_init_pos.cpu().numpy(),
                    quat=self.base_init_quat.cpu().numpy(),
                ),
            )

        self.scene.build(n_envs=num_envs, env_spacing=(1.5, 1.5))

        # --- Motor Mapping ---
        # Genesis maps joints by name. We filter for the actuated joints.
        # ADDED DEBUGGING BLOCK to print available joints if lookup fails
        try:
            self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        except Exception as e:
            print("\n" + "="*50)
            print("ERROR: Joint name not found in the loaded robot model.")
            print("Please update 'dof_names' in your config to match the names below:")
            print("-" * 20)
            # Try to print all joint names available in the entity
            try:
                for j in self.robot.joints:
                    print(f" - {j.name}")
            except:
                print("Could not list joints automatically. Please check your XML/URDF file.")
            print("="*50 + "\n")
            raise e

        # PD Control Setup
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # --- Reward Prep ---
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # --- Buffers ---
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], 1.0, 1.0],
            device=self.device, dtype=gs.tc_float
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # Default Pose Tensor
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        # Jumping State Buffers
        self.jump_toggled_buf = torch.zeros((self.num_envs,), device=self.device)
        self.jump_target_height = torch.zeros((self.num_envs,), device=self.device)
        self.extras = dict()

    # --- Command Sampling (Go2 Logic) ---
    def _sample_commands(self, envs_idx):
        # Sample standard locomotion commands
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["height_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 4] = 0.0 # Default jump command is 0

        # Scale velocities based on height (Go2 heuristic)
        height_diff_scale = 0.5 + torch.abs(self.commands[envs_idx, 3] - self.reward_cfg["base_height_target"]) / \
                            (self.command_cfg["height_range"][1] - self.reward_cfg["base_height_target"]) * 0.5
        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale

    def _sample_jump_commands(self, envs_idx):
        # Specifically toggle jump command
        self.commands[envs_idx, 4] = gs_rand_float(*self.command_cfg["jump_range"], (len(envs_idx),), self.device)

    # --- Main Step ---
    def step(self, actions):
        # Clip and Apply
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # Sim Latency (Use last action if enabled, else current)
        exec_actions = self.last_actions if self.env_cfg["simulate_action_latency"] else self.actions

        # PD Target Calculation
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        # Physics Step
        self.scene.step()

        # --- State Update ---
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()

        # Compute Euler for termination check
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # --- Resampling Logic ---
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._sample_commands(envs_idx)

        # Random injection logic (5% chance)
        random_idxs_1 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
        self._sample_commands(random_idxs_1)
        random_idxs_2 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
        self._sample_jump_commands(random_idxs_2)

        # --- Jump State Machine ---
        jump_cmd_now = (self.commands[:, 4] > 0.0).float()
        toggle_mask = ((self.jump_toggled_buf == 0.0) & (jump_cmd_now > 0.0)).float()

        # Activate jump buffer if toggled
        self.jump_toggled_buf += toggle_mask * self.reward_cfg["jump_reward_steps"]
        # Decay buffer
        self.jump_toggled_buf = torch.clamp(self.jump_toggled_buf - 1.0, min=0.0)
        # Latch target height
        self.jump_target_height = torch.where(jump_cmd_now > 0.0, self.commands[:, 4], self.jump_target_height)

        # --- Termination ---
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # Handle Resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # --- Reward Calculation ---
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # --- Observation Construction ---
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 5
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                (self.jump_toggled_buf / self.reward_cfg["jump_reward_steps"]).unsqueeze(-1),  # 1
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.commands[:, 4] = 0.0 # Reset jump trigger

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, {}

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0: return

        # Reset DOFs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(self.dof_pos[envs_idx], self.motor_dofs, zero_velocity=True, envs_idx=envs_idx)

        # Reset Base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset Internals
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = False # Clear reset flag
        self.jump_toggled_buf[envs_idx] = 0.0

        # Log Rewards
        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)
        self.commands[envs_idx, 3] = self.reward_cfg["base_height_target"] # Enforce default height on reset

    # --- Reward Functions (Go2 Logic) ---
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_pos[:, 2] - self.commands[:, 3])

    def _reward_jump_height_tracking(self):
        mask = ((self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"]) &
                (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"]))
        return mask.float() * torch.exp(-torch.square(self.base_pos[:, 2] - self.jump_target_height))

    def _reward_jump_height_achievement(self):
        mask = ((self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"]) &
                (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"]))
        return mask.float() * (torch.abs(self.base_pos[:, 2] - self.jump_target_height) < 0.2).float()

    def _reward_jump_speed(self):
        mask = ((self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"]) &
                (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"]))
        return mask.float() * torch.exp(self.base_lin_vel[:, 2]) * 0.2

    def _reward_jump_landing(self):
        mask = (self.jump_toggled_buf >= 0.6 * self.reward_cfg["jump_reward_steps"])
        return mask.float() * -torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

class GenesisSB3Wrapper(VecEnv):
    """
    Wraps the Genesis GPU environment to be compatible with Stable Baselines 3.
    """
    def __init__(self, env: SpotGenesisEnv):
        self.env = env

        # Create Spaces
        # Obs: [ang_vel(3), grav(3), cmd(5), dof_pos(12), dof_vel(12), act(12), jump_state(1)] = 48
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.num_obs,), dtype=np.float32)
        act_space = spaces.Box(low=-1, high=1, shape=(self.env.num_actions,), dtype=np.float32)

        super().__init__(env.num_envs, obs_space, act_space)
        self.actions = None

    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions):
        # Store actions, convert to tensor on correct device
        if isinstance(actions, np.ndarray):
            self.actions = torch.from_numpy(actions).to(self.env.device).float()
        else:
            self.actions = actions

    def step_wait(self):
        # Step the GPU simulation
        obs, rews, dones, extras = self.env.step(self.actions)

        # Transfer data back to CPU for SB3
        obs_np = obs.cpu().numpy()
        rews_np = rews.cpu().numpy()
        dones_np = dones.cpu().numpy()

        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones_np[i]:
                infos[i]["terminal_observation"] = obs_np[i]

        return obs_np, rews_np, dones_np, infos

    def close(self):
        pass

    # --- Missing Abstract Methods for VecEnv ---

    def get_attr(self, attr_name, indices=None):
        """Return attribute from underlying environment (broadcasted)."""
        # Since Genesis is monolithic, we assume the attribute is the same for all
        val = getattr(self.env, attr_name)
        return [val] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on underlying environment."""
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Call method on underlying environment."""
        method = getattr(self.env, method_name)
        return [method(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped."""
        return [False] * self.num_envs