import genesis as gs
import torch
import numpy as np
import time
from stable_baselines3 import PPO
from pynput import keyboard
from spot_genesis_env import SpotGenesisEnv, GenesisSB3Wrapper

# --- Configuration (Must match training!) ---
try:
    from robot_descriptions.loaders.mujoco import load_robot_description
    from robot_descriptions import spot_mj_description
    SPOT_XML_PATH = spot_mj_description.MJCF_PATH
except ImportError:
    print("WARNING: robot_descriptions not found. Using local 'urdf/spot/spot.urdf'")
    SPOT_XML_PATH = "urdf/spot/spot.urdf"

class KeyboardController:
    def __init__(self):
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0
        self.cmd_ang_vel = 0.0
        self.cmd_height = 0.5  # Default height
        self.cmd_jump = 0.0
        
        # Ranges
        self.vel_x_scale = 1.0
        self.vel_y_scale = 0.5
        self.ang_vel_scale = 0.8
        self.height_range = [0.4, 0.6]
        
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if hasattr(key, 'char'):
                self.pressed_keys.add(key.char)
            else:
                self.pressed_keys.add(key)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if hasattr(key, 'char'):
                self.pressed_keys.discard(key.char)
            else:
                self.pressed_keys.discard(key)
        except AttributeError:
            pass

    def get_command(self):
        # Reset velocities (WASD style - move only when pressed)
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0
        self.cmd_ang_vel = 0.0
        self.cmd_jump = 0.0

        # Linear X (Forward/Back)
        if 'w' in self.pressed_keys: self.cmd_vel_x += 1.0
        if 's' in self.pressed_keys: self.cmd_vel_x -= 1.0
        
        # Linear Y (Strafing)
        if 'a' in self.pressed_keys: self.cmd_vel_y += 1.0
        if 'd' in self.pressed_keys: self.cmd_vel_y -= 1.0
        
        # Angular (Turning) - using Q/E
        if 'q' in self.pressed_keys: self.cmd_ang_vel += 1.0
        if 'e' in self.pressed_keys: self.cmd_ang_vel -= 1.0

        # Height Control (Arrows) - Incremental
        if keyboard.Key.up in self.pressed_keys:
            self.cmd_height += 0.005
        if keyboard.Key.down in self.pressed_keys:
            self.cmd_height -= 0.005
        self.cmd_height = np.clip(self.cmd_height, self.height_range[0], self.height_range[1])

        # Jump (Space)
        if keyboard.Key.space in self.pressed_keys:
            self.cmd_jump = 0.7  # Target jump height

        return np.array([
            self.cmd_vel_x * self.vel_x_scale,
            self.cmd_vel_y * self.vel_y_scale,
            self.cmd_ang_vel * self.ang_vel_scale,
            self.cmd_height,
            self.cmd_jump
        ], dtype=np.float32)

def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "robot_path": SPOT_XML_PATH,
        "dof_names": [
            "fl_hx", "fl_hy", "fl_kn",
            "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn",
            "hr_hx", "hr_hy", "hr_kn",
        ],
        "default_joint_angles": { 
            "fl_hx": 0.0,  "fl_hy": 0.8,  "fl_kn": -1.5,
            "fr_hx": 0.0,  "fr_hy": 0.8,  "fr_kn": -1.5,
            "hl_hx": 0.0,  "hl_hy": 0.8,  "hl_kn": -1.5,
            "hr_hx": 0.0,  "hr_hy": 0.8,  "hr_kn": -1.5,
        },
        "kp": 50.0, 
        "kd": 1.0,
        "termination_if_roll_greater_than": 0.4,
        "termination_if_pitch_greater_than": 0.4,
        "base_init_pos": [0.0, 0.0, 0.55],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 999999.0, # Disable auto-resampling for teleop
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    
    obs_cfg = { "num_obs": 48, "obs_scales": { "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05 } }
    reward_cfg = { "reward_scales": {}, "base_height_target": 0.50, "tracking_sigma": 0.25, "jump_reward_steps": 50 } 
    command_cfg = { 
        "num_commands": 5, 
        "lin_vel_x_range": [-1.0, 2.0], "lin_vel_y_range": [-0.5, 0.5], "ang_vel_range": [-0.8, 0.8],
        "height_range": [0.4, 0.6], "jump_range": [0.6, 0.8]
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    # 1. Init Genesis (Enable Viewer GUI)
    # CHANGED: Force CPU backend to avoid CUDA driver issues
    gs.init(logging_level="warning", backend=gs.constants.backend.cpu)

    # 2. Config & Env
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create SINGLE env for visualization
    # CHANGED: Force device="cpu" so tensors are not created on the GPU
    env_base = SpotGenesisEnv(
        num_envs=1, 
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True, # <--- ENABLES 3D VIEWER
        device="cpu"
    )
    env = GenesisSB3Wrapper(env_base)

    # 3. Load Model
    # Make sure final_model.zip is in the same folder!
    try:
        # CHANGED: Explicitly load model to CPU
        model = PPO.load("final_model_3.zip", env=env, device="cpu")
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model. Did you download 'final_model_3.zip' from Colab?")
        raise e

    # 4. Init Controller
    print("\n--- KEYBOARD TELEOP ENABLED ---")
    print("Use WASD to move, Q/E to turn.")
    print("Use Up/Down Arrows to adjust height.")
    print("Press SPACE to Jump.")
    print("Press ESC in terminal to stop listener (or Ctrl+C).")
    controller = KeyboardController()

    # 5. Run Loop
    obs = env.reset()
    
    # Scale factors for observation patching (must match obs_scales)
    # [lin_vel (x2), ang_vel, height, jump] -> [2.0, 2.0, 0.25, 1.0, 1.0]
    cmd_scales = np.array([2.0, 2.0, 0.25, 1.0, 1.0], dtype=np.float32)

    while True:
        # A. Get User Command
        user_cmd = controller.get_command()
        
        # B. Inject into Environment (so reward/internal state is correct)
        # Note: env_base.commands is a tensor [num_envs, 5]
        env_base.commands[0, :] = torch.tensor(user_cmd, device=env_base.device)
        
        # C. Patch the Observation (CRITICAL: The policy reacts to the obs, not the internal state)
        # Obs index 6:11 contains the scaled commands
        obs[0, 6:11] = user_cmd * cmd_scales

        # D. Predict & Step
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Slow down loop slightly to make visualization smoother/easier to watch
        time.sleep(0.01)

        if dones[0]:
            print("Resetting...")
            # On reset, we might want to reset the controller height or keep it
            # controller.cmd_height = 0.5 

if __name__ == "__main__":
    main()