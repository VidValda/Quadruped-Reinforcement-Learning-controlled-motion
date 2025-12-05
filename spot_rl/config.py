from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x_range: tuple[float, float] = (-1.0, 2.0)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_range: tuple[float, float] = (-0.6, 0.6)
    height_range: tuple[float, float] = (0.2, 0.4)
    jump_range: tuple[float, float] = (0.5, 1.5)
    resampling_time_s: float = 4.0
    num_commands: int = 5


@dataclass(frozen=True)
class SimulationConfig:
    frame_skip: int = 5
    target_height: float = 0.35
    max_episode_steps: int = 2000
    episode_length_s: float = 20.0
    clip_actions: float = 100.0
    action_scale: float = 0.25
    kp: float = 20.0
    kd: float = 0.5
    termination_if_pitch_greater_than: float = 0.174533
    termination_if_roll_greater_than: float = 0.174533
    simulate_action_latency: bool = True


@dataclass(frozen=True)
class RewardConfig:
    base_height_target: float = 0.3
    tracking_sigma: float = 0.25
    jump_reward_steps: int = 50
    reward_scales: dict = None
    
    def __post_init__(self):
        if self.reward_scales is None:
            object.__setattr__(self, 'reward_scales', {
                'tracking_lin_vel': 1.0,
                'tracking_ang_vel': 0.2,
                'lin_vel_z': -1.0,
                'action_rate': -0.005,
                'similar_to_default': -0.1,
                'base_height': -50.0,
                'jump_height_tracking': 0.5,
                'jump_height_achievement': 10.0,
                'jump_speed': 1.0,
                'jump_landing': 0.08,
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
    n_steps: int = 24
    batch_size: int = 0
    n_epochs: int = 5
    learning_rate: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    policy_kwargs: dict = None
    device: str = "cpu"
    
    def __post_init__(self):
        if self.policy_kwargs is None:
            object.__setattr__(self, 'policy_kwargs', {
                'net_arch': [dict(pi=[512, 256, 128], vf=[512, 256, 128])],
                'activation_fn': 'elu'
            })


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


