from typing import Optional, Dict, Any, Tuple, SupportsFloat

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.core import ObsType, ActType
from gymnasium.envs.registration import EnvSpec
import numpy as np
import pandas as pd

DEFAULT_WALLET = 1e6


class PortfolioEnv(Env):
    action_space: Space[ActType]
    observation_space: Space[ObsType]
    reward_range: Tuple[float, float] = -1, np.inf
    spec: EnvSpec | None = None

    def __init__(self, wallet: float = DEFAULT_WALLET):
        self.wallet = wallet

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Dict[str, Any] = {}
    ) -> tuple[ObsType, Dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info"""
        super().reset(seed=seed)
        if 'wallet' in options:
            self.wallet = options['wallet']

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        pass
