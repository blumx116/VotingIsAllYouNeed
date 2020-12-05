from typing import List, Iterable, Dict, Tuple

import numpy as np



from VIAYN.project_types import (
    Agent, Environment, SystemConfiguration, VotingConfiguration, A, S, B,
    HistoryItem, WeightedBet, ActionBet, Action, PayoutConfiguration, PolicyConfiguration)

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
    config.voting_manager.set_n_agents(len(agents))

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

            action: A = select_action(
                placed_bets,
                config.policy_manager,
                config.voting_manager)

            env.step(action)

def select_action(
        placed_bets: Dict[A, List[WeightedBet[A, S]]],
        policy_config: PolicyConfiguration,
        voting_config: VotingConfiguration) \
        -> A:
    action: A
    for action in placed_bets:
        bet: WeightedBet[A, S]
        for bet in placed_bets[action]:
            policy_config.validate_bet(bet)
            voting_config.validate_bet(bet)

    aggregated_bets: Dict[A, B] = policy_config.aggregate_bets(placed_bets)
    return policy_config.select_action(aggregated_bets)

def calculate_payouts(
        history: List[HistoryItem[A, S]],
        welfare_score: float,
        payout_config: PayoutConfiguration,
        t: int) \
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