# python main.py --mode train
# python main.py --mode teleop

from __future__ import annotations

import argparse

from scripts.teleop import main as teleop_main
from scripts.train import main as train_main


def main():
    parser = argparse.ArgumentParser(description="Spot RL legacy launcher.")
    parser.add_argument(
        "--mode",
        choices=("train", "teleop"),
        default="teleop",
        help="Launch training (train) or human-in-the-loop teleoperation (teleop).",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_main()
    else:
        teleop_main()


if __name__ == "__main__":
    main()