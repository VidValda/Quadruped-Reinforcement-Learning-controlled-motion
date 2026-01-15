from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: tuple[float, float] = (-0.5, 1)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_range: tuple[float, float] = (-0.3, 0.3)
    resampling_time_s: float = 6.0


@dataclass(frozen=True)
class SimulationConfig:
    frame_skip: int = 5
    target_height: float = 0.5247
    max_episode_steps: int = 4000
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.55)


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 15_000_000
    n_steps: int = 1024
    batch_size: int = 4096
    n_epochs: int = 5
    learning_rate: float = 1e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    device: str = "cpu"
    num_envs: Optional[int] = None 


@dataclass(frozen=True)
class PathConfig:
    model_path: Path = ROOT_DIR / "models" / "ppo_spot_v39.zip"
    stats_path: Path = ROOT_DIR / "stats" / "vec_normalize_stats_v39.pkl"
    tensorboard_log: Path = ROOT_DIR / "spot_tensorboard_advanced"


COMMAND = CommandConfig()
SIMULATION = SimulationConfig()
TRAINING = TrainingConfig()
PATHS = PathConfig()


