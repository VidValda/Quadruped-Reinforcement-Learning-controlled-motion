import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import os
import multiprocessing
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from pynput import keyboard

try:
    from robot_descriptions.loaders.mujoco import load_robot_description
    from robot_descriptions import spot_mj_description
except ImportError:
    print("Error: 'robot_descriptions' package not found.")
    print("Please install it: pip install robot-descriptions")
    exit()

# --- Global variables for keyboard command ---
lin_x = 0.0
lin_y = 0.0
ang_z = 0.0
stop = False

# --- Spot Command Ranges (from CustomSpotEnv) ---
LIN_X_RANGE = [-0.5, 1.0]
LIN_Y_RANGE = [-0.3, 0.3]
ANG_Z_RANGE = [-0.5, 0.5]

def on_press(key):
    global lin_x, lin_y, ang_z, stop
    try:
        if key.char == 'w':
            lin_x += 0.1
        elif key.char == 's':
            lin_x -= 0.1
        elif key.char == 'a':
            lin_y += 0.1
        elif key.char == 'd':
            lin_y -= 0.1
        elif key.char == 'q':
            ang_z += 0.1
        elif key.char == 'e':
            ang_z -= 0.1
        elif key.char == '8':
            stop = True
        
        # Clip commands to the environment's supported ranges
        lin_x = np.clip(lin_x, LIN_X_RANGE[0], LIN_X_RANGE[1])
        lin_y = np.clip(lin_y, LIN_Y_RANGE[0], LIN_Y_RANGE[1])
        ang_z = np.clip(ang_z, ANG_Z_RANGE[0], ANG_Z_RANGE[1])
            
        # Clear the console and print current commands
        os.system('clear' if os.name != 'nt' else 'cls')
        print("--- Robot Commands ---")
        print(f"Forward (w/s): {lin_x:.2f} m/s")
        print(f"Strafe (a/d):  {lin_y:.2f} m/s")
        print(f"Turn (q/e):    {ang_z:.2f} rad/s")
        print("\nPress '8' to stop.")

    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

class CustomSpotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = spot_mj_description.MJCF_PATH
        xml_dir = os.path.dirname(xml_path)
        assets_dir = os.path.join(xml_dir, 'assets')
        assets_dir = os.path.abspath(assets_dir).replace('\\', '/')

        # --- XML Modification Section ---
        with open(xml_path, 'r') as f:
            xml_string = f.read()
            
        if 'meshdir="assets"' in xml_string:
            xml_string = xml_string.replace('meshdir="assets"', f'meshdir="{assets_dir}"')
        else:
            print("Warning: Could not find 'meshdir=\"assets\"' in XML. Mesh loading might fail.")
            if "<compiler" not in xml_string:
                xml_string = xml_string.replace("<mujoco model=\"spot\">", f"<mujoco model=\"spot\">\n  <compiler meshdir=\"{assets_dir}\"/>", 1)

        floor_assets = """
<asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
</asset>
"""
        floor_geom = '    <geom name="floor" type="plane" size="10 10 0.1" material="grid"/>'

        if "<asset>" not in xml_string:
            if "</compiler>" in xml_string:
                xml_string = xml_string.replace("</compiler>", f"</compiler>\n{floor_assets}", 1)
            else:
                xml_string = xml_string.replace("<mujoco model=\"spot\">", f"<mujoco model=\"spot\">\n{floor_assets}", 1)
        else:
            asset_additions = """
    <texture type="2d" name="grid" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
"""
            xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_additions}", 1)

        if "<worldbody>" not in xml_string:
            print("Error: <worldbody> not found in model XML. Floor not added.")
            self.model = load_robot_description("spot_mj_description")
        else:
            xml_string = xml_string.replace("<worldbody>", f"<worldbody>\n{floor_geom}", 1)
            
        try:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        except Exception as e:
            print("Error compiling modified XML string:")
            print(e)
            print("Fallback: Loading original model without floor.")
            self.model = load_robot_description("spot_mj_description")
        # --- End XML Modification ---

        self.data = mujoco.MjData(self.model)
        
        self.frame_skip = 5 # --- Tunable ---
        self.dt = self.frame_skip * self.model.opt.timestep
        self.render_mode = render_mode
        self.viewer = None
        
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'body')
        if self.torso_body_id == -1:
            raise ValueError("Could not find body named 'body' in the XML model.")

        # --- Tunable: Default Pose ---
        self.default_homing_pose = np.array([
            0.0, 0.7, -1.4,
            0.0, 0.7, -1.4,
            0.0, 0.7, -1.4,
            0.0, 0.7, -1.4
        ])
        
        self.target_height = 0.35 # --- Tunable ---
        self.last_action = np.zeros(self.model.nu)

        # --- Tunable: Command Ranges ---
        self.command_cfg = {
            "lin_vel_x_range": LIN_X_RANGE,
            "lin_vel_y_range": LIN_Y_RANGE,
            "ang_vel_range": ANG_Z_RANGE,
            "resampling_time_s": 4.0
        }
        self.resampling_steps = int(self.command_cfg["resampling_time_s"] / self.dt)
        self.episode_length_buf = 0
        self.target_lin_vel = np.zeros(2)
        self.target_ang_vel = 0.0
        
        # --- ADDED FOR MANUAL CONTROL ---
        self.manual_control = False

        num_actuators = self.model.nu
        # --- Action Space Definition ---
        self.action_space = spaces.Box(
            low=-0.5, # --- Tunable ---
            high=0.5, # --- Tunable ---
            shape=(num_actuators,), 
            dtype=np.float32
        )
        
        # --- Observation Space Definition ---
        num_joint_pos = self.model.nq - 7
        num_joint_vel = self.model.nv - 6
        num_root_vel = 6 
        num_sensors = 0
        num_commands = 3
        
        total_obs_dim = (
            num_joint_pos + 
            num_joint_vel + 
            num_root_vel + 
            1 +
            4 +
            num_sensors +
            num_commands 
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )

    def _get_obs(self):
        torso_xpos = self.data.body(self.torso_body_id).xpos
        torso_quat = self.data.body(self.torso_body_id).xquat
        torso_z_pos = torso_xpos[2]

        # --- Observation Vector ---
        return np.concatenate([
            self.data.qpos[7:],    
            self.data.qvel[6:],    
            self.data.qvel[0:6],   
            np.array([torso_z_pos]), 
            torso_quat,
            self.target_lin_vel,
            np.array([self.target_ang_vel])
        ]).astype(np.float32)

    def _quat_to_roll_pitch(self, quat):
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

    def _resample_commands(self):
        # --- Command Sampling ---
        # --- MODIFIED: Don't resample if under manual control ---
        if self.manual_control:
            return
            
        self.target_lin_vel[0] = self.np_random.uniform(*self.command_cfg["lin_vel_x_range"])
        self.target_lin_vel[1] = self.np_random.uniform(*self.command_cfg["lin_vel_y_range"])
        self.target_ang_vel = self.np_random.uniform(*self.command_cfg["ang_vel_range"])

    # --- ADDED: Method to enable manual control ---
    def enable_manual_control(self):
        """Disables automatic command resampling."""
        self.manual_control = True
        print("Manual control enabled.")

    # --- ADDED: Method to set commands externally ---
    def set_target_velocities(self, lin_vel, ang_vel):
        """
        Sets the target velocities for the policy.
        lin_vel: [vx, vy]
        ang_vel: wz
        """
        self.target_lin_vel[0] = lin_vel[0]
        self.target_lin_vel[1] = lin_vel[1]
        self.target_ang_vel = ang_vel

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # --- Reset Simulation ---
        mujoco.mj_resetData(self.model, self.data)
        
        self.last_action = np.zeros(self.model.nu)
        self.episode_length_buf = 0
        
        # --- MODIFIED: Only resample if not in manual mode ---
        if not self.manual_control:
            self._resample_commands()
        
        obs = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info

    def step(self, action):
        # --- Action Application ---
        action = np.clip(action, self.action_space.low, self.action_space.high)
        final_action = self.default_homing_pose + action
        
        final_action_clipped = np.clip(final_action, -2*np.pi, 2*np.pi)
        self.data.ctrl[:] = final_action_clipped
        
        # --- Simulation Step ---
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.episode_length_buf += 1
        
        # --- MODIFIED: Only resample if not in manual mode ---
        if self.episode_length_buf % self.resampling_steps == 0 and not self.manual_control:
            self._resample_commands()
        
        obs = self._get_obs()

        # --- Reward Calculation ---
        current_lin_vel = self.data.body(self.torso_body_id).cvel[3:5]
        current_ang_vel = self.data.body(self.torso_body_id).cvel[2]
        
        lin_vel_error = np.linalg.norm(self.target_lin_vel - current_lin_vel)
        ang_vel_error = np.square(self.target_ang_vel - current_ang_vel)
        
        lin_vel_reward = np.exp(-1.5 * lin_vel_error) # --- Tunable: Reward Weight ---
        ang_vel_reward = np.exp(-1.0 * ang_vel_error) # --- Tunable: Reward Weight ---
        
        torso_z_pos = self.data.body(self.torso_body_id).xpos[2]
        torso_quat = self.data.body(self.torso_body_id).xquat
        roll, pitch = self._quat_to_roll_pitch(torso_quat)
        
        height_penalty = np.square(torso_z_pos - self.target_height)
        orientation_penalty = np.square(roll) + np.square(pitch)

        action_rate_penalty = np.sum(np.square(action - self.last_action))
        control_cost = np.sum(np.square(action))

        self.last_action = action
        
        # --- Total Reward (Tunable Weights) ---
        reward = (
            2.0 * lin_vel_reward +
            1.0 * ang_vel_reward -
            2.0 * height_penalty -
            1.0 * orientation_penalty -
            0.1 * action_rate_penalty -
            0.03 * control_cost
        )
        
        # --- Termination Condition ---
        terminated = torso_z_pos < 0.2
        
        if terminated:
            reward = -10.0 # --- Tunable: Fall Penalty ---

        truncated = False
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def make_env(render_mode=None):
    env = CustomSpotEnv(render_mode=render_mode)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) # --- Tunable: Max Steps ---
    return env

def main():
    global lin_x, lin_y, ang_z, stop
    
    # --- Main Toggles ---
    # Set this to True to train a new model
    TRAIN = False
    
    # --- Training Config ---
    TOTAL_TIMESTEPS = 30_000_000
    MODEL_PATH = "ppo_spot_v10.zip"
    STATS_PATH = "vec_normalize_stats_v10.pkl"
    
    # --- PPO Hyperparameters (Tunable) ---
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    LR = 3e-4
    GAMMA = 0.99
    DEVICE = "cuda" # "cpu" or "cuda"

    if TRAIN:
        num_cpu = multiprocessing.cpu_count()
        print(f"--- STARTING TRAINING (using {num_cpu} processes) ---")
        
        print("Creating vectorized environment for training...")
        env_fns = [lambda: make_env(render_mode=None) for _ in range(num_cpu)]
        env = SubprocVecEnv(env_fns)
        
        print("Wrapping environment with VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=GAMMA)

        # --- Model Definition ---
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./spot_tensorboard_advanced/",
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LR,
            gamma=GAMMA,
            device=DEVICE
        )
        
        print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        
        print(f"Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
        
        print(f"Saving VecNormalize stats to {STATS_PATH}")
        env.save(STATS_PATH)
        
        print(f"--- TRAINING COMPLETE ---")
        env.close() 

    else:
        # --- MODIFIED: EVALUATION WITH KEYBOARD ---
        print(f"--- STARTING KEYBOARD VISUALIZATION ---")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
            print(f"Error: Model file '{MODEL_PATH}' or stats file '{STATS_PATH}' not found.")
            print("Set TRAIN = True to train a model first.")
            exit()
            
        print("--- MODEL AND STATS FOUND ---")
        
        # Start keyboard listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        vis_env_base = DummyVecEnv([lambda: make_env(render_mode="human")])
        
        print(f"Loading VecNormalize stats from {STATS_PATH}...")
        vis_env = VecNormalize.load(STATS_PATH, vis_env_base)
        
        vis_env.training = False
        vis_env.norm_reward = False
        
        print(f"Loading model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=vis_env, device=DEVICE)
        print("--- MODEL LOADED ---")

        # Tell environment to listen for manual commands
        vis_env.env_method('enable_manual_control')

        on_press(None) # Print initial command state
        obs = vis_env.reset()
        
        # --- Visualization Loop (controlled by '8' key) ---
        while not stop:
            # Set target velocities from global keyboard vars
            # env_method calls the function on the underlying env
            vis_env.env_method('set_target_velocities', [lin_x, lin_y], ang_z)
            
            action, _states = model.predict(obs, deterministic=True)
            vec_obs, vec_reward, vec_terminated, vec_info = vis_env.step(action)
            
            obs = vec_obs
            info = vec_info[0]
            terminated = vec_terminated[0]
            truncated = info.get('TimeLimit.truncated', False)

            if terminated or truncated:
                print("--- Episode Finished, Resetting Environment ---")
                obs = vis_env.reset()
                on_press(None) # Re-print command state after reset

        vis_env.close()
        listener.stop()
        print("--- VISUALIZATION STOPPED ---")
    
if __name__ == "__main__":
    main()
