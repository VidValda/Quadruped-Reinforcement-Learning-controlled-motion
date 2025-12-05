from __future__ import annotations

import sys
from pathlib import Path

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

    vis_env.env_method("enable_manual_control")
    obs = vis_env.reset()

    while not keyboard_controller.state.stop:
        vis_env.env_method(
            "set_target_velocities",
            [keyboard_controller.state.lin_x, keyboard_controller.state.lin_y],
            keyboard_controller.state.ang_z,
            keyboard_controller.state.height,
            keyboard_controller.state.jump,
        )

        action, _ = model.predict(obs, deterministic=True)
        vec_obs, _, vec_dones, vec_infos = vis_env.step(action)

        obs = vec_obs
        info = vec_infos[0]
        terminated = vec_dones[0]
        truncated = info.get("TimeLimit.truncated", False)

        if terminated or truncated:
            obs = vis_env.reset()
            keyboard_controller.state.print_status()

    vis_env.close()
    keyboard_controller.stop()
    print("--- VISUALISATION STOPPED ---")


if __name__ == "__main__":
    main()


