"""Safe, offline reinforcement learning (contextual bandit) - advisory only."""
from .advisor import advise, load_model
from .bandit import ContextualBandit
from .env import RewardConfig
from .evaluate import run_experiment
from .strategies import allowed_strategies, apply_strategy, validate_actions
