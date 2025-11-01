import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from robot_descriptions.loaders.mujoco import load_robot_description
    from robot_descriptions import spot_mj_description
except ImportError:
    print("Error: 'robot_descriptions' package not found.")
    print("Please install it: pip install robot-descriptions")
    exit()

class CustomSpotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = spot_mj_description.MJCF_PATH
        xml_dir = os.path.dirname(xml_path)
        assets_dir = os.path.join(xml_dir, 'assets')
        assets_dir = os.path.abspath(assets_dir)
        assets_dir = assets_dir.replace('\\', '/')

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
            print("--- XML String (first 1000 chars) ---")
            print(xml_string[:1000])
            print("------------------")
            print("Fallback: Loading original model without floor.")
            self.model = load_robot_description("spot_mj_description")

        self.data = mujoco.MjData(self.model)
        
        self.frame_skip = 5
        self.render_mode = render_mode
        self.viewer = None
        
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'body')
        if self.torso_body_id == -1:
            raise ValueError("Could not find body named 'body' in the XML model.")

        num_actuators = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(num_actuators,), 
            dtype=np.float32
        )
        
        num_joint_pos = self.model.nq - 7
        num_joint_vel = self.model.nv - 6
        num_sensors = 0
        obs_shape = (num_joint_pos + num_joint_vel + 1 + 1 + 4 + num_sensors,)
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=obs_shape, 
            dtype=np.float32
        )

    def _get_obs(self):
        torso_xpos = self.data.body(self.torso_body_id).xpos
        torso_quat = self.data.body(self.torso_body_id).xquat
        torso_x_vel = self.data.body(self.torso_body_id).cvel[3] 
        torso_z_pos = torso_xpos[2]

        return np.concatenate([
            self.data.qpos[7:],
            self.data.qvel[6:],
            np.array([torso_x_vel]),
            np.array([torso_z_pos]),
            torso_quat,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        
        obs = self._get_obs()

        forward_velocity = self.data.body(self.torso_body_id).cvel[3]
        control_cost = 0.01 * np.sum(np.square(action))
        healthy_reward = 0 
        reward = forward_velocity + healthy_reward - control_cost
        
        torso_z_position = self.data.body(self.torso_body_id).xpos[2]
        terminated = torso_z_position < 0.2
        
        if terminated:
            reward = -10.0

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

if __name__ == "__main__":
    
    TRAIN = False
    TOTAL_TIMESTEPS = 1_000_000
    MODEL_PATH = "ppo_spot_v3.zip"
    
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    LEARNING_RATE = 3e-4
    GAMMA = 0.99 

    def make_env(render_mode=None):
        env = CustomSpotEnv(render_mode=render_mode)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env

    if TRAIN:
        print("--- STARTING TRAINING ---")
        print("Creating environment for training...")
        env = DummyVecEnv([make_env])

        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./spot_tensorboard/",
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA
        )
        
        print("Starting training...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        print(f"--- TRAINING COMPLETE, MODEL SAVED TO {MODEL_PATH} ---")
        env.close()

    else:
        print(f"--- SKIPPING TRAINING, LOADING MODEL FROM {MODEL_PATH} ---")
        if not os.path.exists(MODEL_PATH):
            print("Error: Model file not found. Set TRAIN = True to train a model first.")
            exit()
            
        model = PPO.load(MODEL_PATH)
        print("--- MODEL LOADED ---")

    print("--- STARTING VISUALIZATION ---")
    vis_env = make_env(render_mode="human")

    for episode in range(5):
        obs, info = vis_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vis_env.step(action)
            episode_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    vis_env.close()
    print("--- VISUALIZATION COMPLETE ---")
