"""
Simple script to test loading and using a trained model.
This script can test both MuJoCo and Genesis-trained models.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from spot_rl.config import PATHS, TRAINING
from spot_rl.envs.spot_env import make_env
from stable_baselines3.common.vec_env import DummyVecEnv


def inspect_model_spaces(model_path):
    """Inspect the model's observation and action spaces without loading with an env."""
    try:
        # Load model without environment to inspect its spaces
        model = PPO.load(str(model_path), device=TRAINING.device)
        return model.observation_space, model.action_space
    except Exception as e:
        print(f"  Could not inspect model: {e}")
        return None, None


def test_model_loading():
    """Test loading the model from the configured path."""
    model_path = PATHS.model_path
    
    print(f"Attempting to load model from: {model_path}")
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print(f"Please check that the file exists or update spot_rl/config.py")
        return None, None
    
    print(f"Model file found! Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # First, inspect the model to determine if it's Genesis-trained
    print("\n--- Inspecting model spaces ---")
    model_obs_space, model_act_space = inspect_model_spaces(model_path)
    
    if model_obs_space is not None and model_act_space is not None:
        print(f"  Model observation space: {model_obs_space}")
        print(f"  Model action space: {model_act_space}")
        
        # Check if this looks like a Genesis model
        is_genesis = (
            hasattr(model_act_space, 'low') and 
            hasattr(model_act_space, 'high') and
            model_act_space.low[0] == -1.0 and 
            model_act_space.high[0] == 1.0
        )
        
        if is_genesis:
            print("  → Detected Genesis-trained model (action space [-1, 1])")
            return try_load_genesis_model(model_path)
        else:
            print("  → Detected MuJoCo-trained model (action space [-100, 100])")
    
    # Try loading with MuJoCo environment (for models with VecNormalize)
    try:
        print("\n--- Attempting to load with MuJoCo + VecNormalize stats ---")
        from spot_rl.training.pipeline import build_visualization_env
        
        if PATHS.stats_path.exists():
            env = build_visualization_env()
            model = PPO.load(str(model_path), env=env, device=TRAINING.device)
            print("✓ Successfully loaded with VecNormalize stats")
            return model, env
        else:
            print("  VecNormalize stats not found, trying without...")
    except Exception as e:
        print(f"  Could not load with stats: {e}")
        print("  Trying without VecNormalize stats...")
    
    # Try loading without VecNormalize (for Genesis-trained models)
    try:
        print("\n--- Attempting to load with MuJoCo (no VecNormalize) ---")
        env_base = DummyVecEnv([lambda: make_env(render_mode="human")])
        model = PPO.load(str(model_path), env=env_base, device=TRAINING.device)
        print("✓ Successfully loaded without VecNormalize stats")
        return model, env_base
    except Exception as e:
        print(f"  Could not load with MuJoCo: {e}")
        print("  Trying with Genesis environment...")
        return try_load_genesis_model(model_path)


def try_load_genesis_model(model_path):
    """Try loading the model with Genesis environment."""
    try:
        print("\n--- Attempting to load with Genesis environment ---")
        import genesis as gs
        from spot_genesis_env import SpotGenesisEnv, GenesisSB3Wrapper
        
        # Get configs - import here to avoid circular dependency
        import sys
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        scripts_path = str(PROJECT_ROOT / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        from train_genesis import get_genesis_configs
        
        # Initialize Genesis
        gs.init(logging_level="warning", backend=gs.constants.backend.cpu)
        
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_genesis_configs()
        
        # Create Genesis environment
        env_base = SpotGenesisEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,  # Disable viewer for testing
            device="cpu"
        )
        env = GenesisSB3Wrapper(env_base)
        
        # Load model
        model = PPO.load(str(model_path), env=env, device=TRAINING.device)
        print("✓ Successfully loaded with Genesis environment")
        return model, env
    except ImportError:
        print("  ERROR: Genesis not available. Install with: pip install genesis-world")
        return None, None
    except Exception as e:
        print(f"  ERROR: Could not load with Genesis: {e}")
        return None, None


def test_model_inference(model, env):
    """Test running a few inference steps with the model."""
    if model is None or env is None:
        print("Cannot test inference: model or env is None")
        return
    
    print("\n--- Testing Model Inference ---")
    print("Running 10 steps with random observations...")
    
    obs = env.reset()
    
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        print(f"Step {i+1}: reward={rewards[0]:.4f}, done={dones[0]}")
        
        if dones[0]:
            print("  Episode ended, resetting...")
            obs = env.reset()
    
    print("✓ Inference test completed successfully!")
    env.close()


def main():
    print("=" * 60)
    print("Model Loading Test Script")
    print("=" * 60)
    print(f"\nModel path: {PATHS.model_path}")
    print(f"Stats path: {PATHS.stats_path}")
    print(f"Device: {TRAINING.device}\n")
    
    model, env = test_model_loading()
    
    if model is not None:
        print(f"\n✓ Model loaded successfully!")
        print(f"  Observation space: {model.observation_space}")
        print(f"  Action space: {model.action_space}")
        
        # Ask user if they want to test inference
        try:
            response = input("\nTest inference? (y/n): ").strip().lower()
            if response == 'y':
                test_model_inference(model, env)
        except KeyboardInterrupt:
            print("\n\nTest cancelled by user.")
        
        print("\n" + "=" * 60)
        print("Model is ready to use!")
        print("You can now use it with:")
        print("  - scripts/teleop.py (for keyboard control)")
        print("  - Or load it in your own scripts")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Failed to load model. Please check:")
        print("  1. The model file exists at the configured path")
        print("  2. The model was trained with compatible settings")
        print("  3. All dependencies are installed")
        print("=" * 60)


if __name__ == "__main__":
    main()

