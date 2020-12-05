from typing import List

import numpy as np

from VIAYN.project_types import (
    Agent, Environment, SystemConfiguration,
    StateType, ActionType, BetAggregationType)

def train(
        agents: List[Agent[ActionType, StateType]],
        env: Environment[ActionType, StateType],
        episode_seeds: List[int],
        config: SystemConfiguration[ActionType, BetAggregationType],
        n_tsteps: int = np.inf):
    pass
