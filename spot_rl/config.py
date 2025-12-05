from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_range: tuple[float, float] = (-0.5, 0.5)
    height_range: tuple[float, float] = (0.25, 0.45)
    jump_range: tuple[float, float] = (0.5, 0.8)
    resampling_time_s: float = 4.0
    num_commands: int = 5


@dataclass(frozen=True)
class SimulationConfig:
    frame_skip: int = 5
    target_height: float = 0.35
    max_episode_steps: int = 2000
    episode_length_s: float = 20.0
    clip_actions: float = 0.5
    action_scale: float = 0.25
    kp: float = 20.0
    kd: float = 0.5
    termination_if_pitch_greater_than: float = 0.5
    termination_if_roll_greater_than: float = 0.5
    simulate_action_latency: bool = True


@dataclass(frozen=True)
class RewardConfig:
    base_height_target: float = 0.35
    tracking_sigma: float = 0.25
    jump_reward_steps: int = 30
    reward_scales: dict = None
    
    def __post_init__(self):
        if self.reward_scales is None:
            object.__setattr__(self, 'reward_scales', {
                'tracking_lin_vel': 1.0,
                'tracking_ang_vel': 0.5,
                'lin_vel_z': 2.0,
                'action_rate': 0.01,
                'similar_to_default': 0.1,
                'base_height': 1.0,
                'jump_height_tracking': 1.0,
                'jump_height_achievement': 2.0,
                'jump_speed': 0.5,
                'jump_landing': 0.5,
            })


@dataclass(frozen=True)
class ObservationConfig:
    num_obs: int = 48
    obs_scales: dict = None
    
    def __post_init__(self):
        if self.obs_scales is None:
            object.__setattr__(self, 'obs_scales', {
                'lin_vel': 2.0,
                'ang_vel': 0.25,
                'dof_pos': 1.0,
                'dof_vel': 0.05,
            })


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
REWARD = RewardConfig()
OBS = ObservationConfig()
TRAINING = TrainingConfig()
PATHS = PathConfig()


