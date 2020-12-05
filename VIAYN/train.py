from typing import List, Iterable, Dict, Tuple

import numpy as np



from VIAYN.project_types import (
    Agent, Environment, SystemConfiguration, VotingConfiguration,
    HistoryItem, WeightedBet, ActionBet, Action, PayoutConfiguration)

A = TypeVar("A")
S = TypeVar("S")
B = TypeVar("B")

def train(
        agents: List[Agent[A, S]],
        env: Environment[A, S],
        episode_seeds: Iterable[int],
        config: SystemConfiguration[A, B, S],
        tsteps_per_episode: int = np.inf):
    old_episode_history: List[List[HistoryItem[A, S]]] = []
    current_history: List[HistoryItem[A, S]] = []
    balances: Dict[Agent[A, S], float] = \
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
            state: S = env.state()

            welfare_score: float = get_agent_votes(
                agents=agents,
                state=state,
                voting_config=config.voting_manager)

            payouts: Dict[Agent[A, S], float] = calculate_payouts(
                current_history,
                welfare_score,
                config.payout_manager)

            agent: Agent[A, S]
            for agent in payouts:
                balances[agent] += payouts[agent]

            placed_bets: Dict[A, List[WeightedBet[A, S]]] = \
                get_agent_bets(agents, balances, env.state(), env.actions())

def calculate_payouts(
        history: List[HistoryItem[A, S]],
        welfare_score: float,
        payout_config: PayoutConfiguration) \
        -> Dict[Agent[A, S], float]:
    ...

def get_agent_votes(
        agents: List[Agent[A, S]],
        state: S,
        voting_config: VotingConfiguration) -> float:
    votes: List[float] = [agent.vote(state) for agent in agents]
    named_votes: List[Tuple[Agent[A, S], float]] = list(zip(agents, votes))
    agent: Agent[A, S]
    vote: float
    for agent, vote in named_votes:
        assert voting_config.vote_range.contains(vote), \
            f"{agent} made an invalid vote {vote} for rule {voting_config.vote_range}"

    return voting_config.aggregate_votes(votes)

def get_agent_bets(
        agents: List[Agent[A, S]],
        balances: Dict[Agent[A, S], float],
        state: S,
        actions: Iterable[A]) \
        -> Dict[A, List[WeightedBet[A, S]]]:
    placed_bets: Dict[A, List[WeightedBet[A, S]]] = {}
    for action in actions:
        placed_bets[action] = []

        for agent in agents:
            money: float = balances[agent]
            bet: ActionBet = agent.bet(state, action, money)
            wbet: WeightedBet[A, S] = WeightedBet(
                bet=bet.bet,
                prediction=bet.prediction,
                action=action,
                money=money,
                cast_by=agent)
            placed_bets[action].append(wbet)

    return placed_bets