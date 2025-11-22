from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spot_rl.teleop.keyboard import HAS_PYNPUT, KeyboardController
from spot_rl.training.pipeline import load_policy_for_teleop


def main():
    if not HAS_PYNPUT:
        print("Error: pynput not installed. Install it via `pip install pynput` to use teleop mode.")
        return

    keyboard_controller = KeyboardController()
    keyboard_controller.start()

    model, vis_env = load_policy_for_teleop()
    print("--- MODEL AND NORMALISATION STATS LOADED ---")

    obs = vis_env.reset()
    keyboard_controller.state.lin_x = 0.0
    keyboard_controller.state.lin_y = 0.0
    keyboard_controller.state.ang_z = 0.0
    
    # Force initial render
    print(" Iniciando visualización...")
    vis_env.envs[0].render()
    time.sleep(2)  # Give time for window to open

    step_count = 0
    try:
        while not keyboard_controller.state.stop:
            # Update manual commands in environment
            vis_env.env_method(
                "update_manual_commands",
                keyboard_controller.state.lin_x,
                keyboard_controller.state.lin_y,
                keyboard_controller.state.ang_z,
            )
            
            if keyboard_controller.state.manual_mode:
                vis_env.env_method("enable_manual_control")
                action = np.zeros(vis_env.action_space.shape)
                
            else:
                vis_env.env_method("disable_manual_control")
                vis_env.env_method(
                    "set_target_velocities",
                    [keyboard_controller.state.lin_x, keyboard_controller.state.lin_y],
                    keyboard_controller.state.ang_z,
                )
                action, _ = model.predict(obs, deterministic=True)
                
                if step_count % 50 == 0:  # Show every 50 steps
                    env = vis_env.envs[0].env if hasattr(vis_env.envs[0], 'env') else vis_env.envs[0]
                    current_lin_vel = env.data.body(env.torso_body_id).cvel[3:5]
                    current_ang_vel = env.data.body(env.torso_body_id).cvel[2]


            vec_obs, _, vec_dones, vec_infos = vis_env.step(action)

            obs = vec_obs
            info = vec_infos[0]
            terminated = vec_dones[0]
            truncated = info.get("TimeLimit.truncated", False)

            if terminated or truncated:
                obs = vis_env.reset()
                print(" Episodio terminado - Reiniciando...")

            step_count += 1
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(" Detenido por usuario (Ctrl+C)")

    finally:
        print(" Cerrando entorno y listener...")
        keyboard_controller.stop()
        vis_env.close()
        print("--- VISUALISATION STOPPED ---")


if __name__ == "__main__":
    main()


