import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import os
import multiprocessing 

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

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
        assets_dir = os.path.abspath(assets_dir).replace('\\', '/')

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
        num_root_vel = 6 
        num_sensors = 0
        obs_shape = (num_joint_pos + num_joint_vel + num_root_vel + 1 + 4 + num_sensors,)
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=obs_shape, 
            dtype=np.float32
        )

    def _get_obs(self):
        torso_xpos = self.data.body(self.torso_body_id).xpos
        torso_quat = self.data.body(self.torso_body_id).xquat
        torso_z_pos = torso_xpos[2]

        return np.concatenate([
            self.data.qpos[7:],    
            self.data.qvel[6:],    
            self.data.qvel[0:6],   
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
        healthy_reward = 1 
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
    MODEL_PATH = "ppo_spot_v6.zip"
    STATS_PATH = "vec_normalize_stats_v6.pkl"
    
    N_STEPS = 2048
    BATCH_SIZE = 32
    N_EPOCHS = 10
    LEARNING_RATE = 3e-4
    GAMMA = 0.99 

    def make_env(render_mode=None):
        env = CustomSpotEnv(render_mode=render_mode)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) 
        return env

    if TRAIN:
        num_cpu = multiprocessing.cpu_count()
        print(f"--- STARTING TRAINING (using {num_cpu} processes) ---")
        
        print("Creating vectorized environment for training...")
        env_fns = [lambda: make_env(render_mode=None) for _ in range(num_cpu)]
        env = SubprocVecEnv(env_fns)
        
        print("Wrapping environment with VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=GAMMA)

        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./spot_tensorboard/",
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            device="cpu"
        )
        
        print("Starting training...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        
        print(f"Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
        
        print(f"Saving VecNormalize stats to {STATS_PATH}")
        env.save(STATS_PATH)
        
        print(f"--- TRAINING COMPLETE ---")
        env.close() 

    else:
        print(f"--- SKIPPING TRAINING, LOADING MODEL FROM {MODEL_PATH} ---")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
            print(f"Error: Model file '{MODEL_PATH}' or stats file '{STATS_PATH}' not found.")
            print("Set TRAIN = True to train a model first.")
            exit()
            
        print("--- MODEL AND STATS FOUND ---")

    print("--- STARTING VISUALIZATION ---")
    
    vis_env_base = DummyVecEnv([lambda: make_env(render_mode="human")])
    
    print(f"Loading VecNormalize stats from {STATS_PATH}...")
    vis_env = VecNormalize.load(STATS_PATH, vis_env_base)
    
    vis_env.training = False
    vis_env.norm_reward = False
    
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, env=vis_env,device="cpu")
    print("--- MODEL LOADED ---")


    for episode in range(5):
        obs = vis_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            vec_obs, vec_reward, vec_terminated, vec_info = vis_env.step(action)
            
            obs = vec_obs
            reward = vec_reward[0]
            terminated = vec_terminated[0]
            info = vec_info[0]
            truncated = info.get('TimeLimit.truncated', False)
            episode_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    vis_env.close()
    print("--- VISUALIZATION COMPLETE ---")
