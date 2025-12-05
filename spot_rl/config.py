from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_range: tuple[float, float] = (-0.5, 0.5)
    resampling_time_s: float = 4.0


@dataclass(frozen=True)
class SimulationConfig:
    frame_skip: int = 5
    target_height: float = 0.35
    max_episode_steps: int = 2000


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 10_000_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    device: str = "cpu"


@dataclass(frozen=True)
class PathConfig:
    model_path: Path = ROOT_DIR / "models" / "ppo_spot_v16.zip"
    stats_path: Path = ROOT_DIR / "stats" / "vec_normalize_stats_v16.pkl"
    tensorboard_log: Path = ROOT_DIR / "spot_tensorboard_advanced"


COMMAND = CommandConfig()
SIMULATION = SimulationConfig()
TRAINING = TrainingConfig()
PATHS = PathConfig()


