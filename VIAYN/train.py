from typing import List, Iterable, Dict

import numpy as np

from VIAYN.project_types import (
    Agent, Environment, SystemConfiguration, HistoryItem, WeightedBet, ActionBet,
    StateType, ActionType, BetAggregationType)

def train(
        agents: List[Agent[ActionType, StateType]],
        env: Environment[ActionType, StateType],
        episode_seeds: Iterable[int],
        config: SystemConfiguration[ActionType, BetAggregationType],
        tsteps_per_episode: int = np.inf):
    old_episode_history: List[List[HistoryItem[ActionType, StateType]]] = []
    current_history: List[HistoryItem[ActionType, StateType]] = []
    balances: Dict[Agent[ActionType, StateType], float] = \
        {agent: 1. for agent in agents}

    seed: int
    for seed in episode_seeds:
        env.reset()
        env.seed(seed)

        if len(current_history) > 0:
            old_episode_history.append(current_history)
        current_history = []

        t: int = 0
        while not env.done() and t < tsteps_per_episode:
            state: StateType = env.state()

            action: ActionType
            placed_bets: Dict[ActionType, List[WeightedBet[ActionType]]] = \
                get_agent_bets(agents, balances, env.state(), env.actions())


def get_agent_bets(
        agents: List[Agent[ActionType, StateType]],
        balances: Dict[Agent[ActionType, StateType], float],
        state: StateType,
        actions: List[ActionType]):
    placed_bets: Dict[ActionType, List[WeightedBet[ActionType]]] = {}
    for action in actions:
        placed_bets[action] = []

        for agent in agents:
            money: float = balances[agent]
            bet: ActionBet = agent.bet(state, action, money)
            wbet: WeightedBet[ActionType, StateType] = WeightedBet(
                bet=bet.bet,
                prediction=bet.prediction,
                action=action,
                money=money,
                cast_by=agent)
            placed_bets[action].append(wbet)

    return placed_bets




