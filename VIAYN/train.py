# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-12-10 14:37:06
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:58:42
from typing import List, Iterable, Dict, Tuple

import numpy as np

from VIAYN.project_types import (
    Agent, Environment, SystemConfiguration, VotingConfiguration, A, S, B, VoteRange,
    HistoryItem, WeightedBet, ActionBet, Action, PayoutConfiguration, PolicyConfiguration)
from VIAYN.utils import add_dictionaries

def train(
        agents: List[Agent[A, S]],
        env: Environment[A, S],
        episode_seeds: Iterable[int],
        config: SystemConfiguration[A, B, S],
        tsteps_per_episode: int = np.inf) \
        -> Tuple[List[List[HistoryItem[A, S]]], Dict[Agent[A, S], float]]:
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
                config=config)

            payouts: Dict[Agent[A, S], float] = calculate_payouts(
                current_history,
                welfare_score,
                config, t)

            agent: Agent[A, S]
            for agent in payouts:
                balances[agent] += payouts[agent]

            placed_bets: Dict[A, List[WeightedBet[A, S]]] = \
                get_agent_bets(agents, balances, env.state(), env.actions())

            action: A = select_action(
                placed_bets,
                config)

            # TODO: make this its own function?
            bets_that_happened: List[WeightedBet[A, S]] = placed_bets[action]
            bet: WeightedBet[A, S]
            for bet in bets_that_happened:
                # duplicate assert
                assert sum(bet.bet) <= 1
                balances[bet.cast_by] *= (1 - sum(bet.bet))

            env.step(action)
            
            current_history.append(
                HistoryItem(
                    selected_action=action,
                    predictions=placed_bets,
                    t_enacted=t))
            t += 1
        
        final_payouts: Dict[Agent[A, S], float] = pay_outstanding_bets(
            current_history,t, config)

        for agent in final_payouts:
            balances[agent] += final_payouts[agent]


        
    old_episode_history.append(current_history)
    # need to append the last one (TODO: redo this logic to avoid duplication)

    return old_episode_history, balances

def pay_outstanding_bets(
        history: List[HistoryItem[A, S]],
        last_t: int,
        config: SystemConfiguration[A, B, S]) -> Dict[Agent[A, S], float]:

    item: HistoryItem[A, S]
    total_payouts: Dict[Agent[A, S], float] = {}
    for item in history:
        bets_that_matter: List[WeightedBet[A, S]] = \
            item.predictions[item.selected_action]
        t_idx: int = last_t - item.t_enacted - 1
        if len(bets_that_matter) == 0:
            continue
        prediction_len: int = len(bets_that_matter[0].prediction)
        delta_t: int
        for delta_t in range(prediction_len - t_idx):
            payouts_for_this_timestep: Dict[Agent[A, S], float] = \
                config.payout_manager.calculate_all_payouts(
                    record=item,
                    welfare_score=0,
                    t_current=last_t + delta_t)
            
            agent: Agent[A, S]
            for agent in payouts_for_this_timestep:
                if agent not in total_payouts:
                    total_payouts[agent] = 0.
                total_payouts[agent] += payouts_for_this_timestep[agent]
                
    return total_payouts



def select_action(
        bets: Dict[A, List[WeightedBet[A, S]]],
        config: SystemConfiguration[A, B, S]) -> A:
    aggregation: Dict[A, B] = config.policy_manager.aggregate_bets(bets)
    return config.policy_manager.select_action(aggregation)



def calculate_payouts(
        history: List[HistoryItem[A, S]],
        welfare_score: float,
        config: SystemConfiguration[A, B, S],
        t: int) \
        -> Dict[Agent[A, S], float]:
    total_payouts: Dict[Agent[A, S], float] = {}
    record: HistoryItem[A, S]
    for record in history:
        payout: Dict[Agent[A, S], float] = \
            config.payout_manager.calculate_all_payouts(
                record=record, welfare_score=welfare_score,
                t_current=t)
        # TODO: poliferate the use of add_dictionaries throughout  train.py
        total_payouts = add_dictionaries(total_payouts, payout)

    return total_payouts


def get_agent_votes(
        agents: List[Agent[A, S]],
        state: S,
        config: SystemConfiguration[A, B, S]) -> float:
    votes: List[float] = [agent.vote(state) for agent in agents]
    vote_range: VoteRange = config.voting_manager.vote_range
    votes = [vote for vote in votes if vote_range.contains(vote)]
    # filter to only valid values
    # TODO : log invalid votes
    return config.voting_manager.aggregate_votes(votes)


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
