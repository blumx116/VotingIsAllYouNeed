# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-12-10 14:37:06
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:58:42
from dataclasses import dataclass
from typing import Generator, List, Iterable, Dict, Tuple, Generic

import numpy as np

from VIAYN.project_types import (
    Agent, Environment, SystemConfiguration, VotingConfiguration, A, S, B, VoteRange,
    HistoryItem, WeightedBet, ActionBet, Action, PayoutConfiguration, PolicyConfiguration,
    AnonymizedHistoryItem)
from VIAYN.utils import add_dictionaries


"""
This file handles the main loop for the betting process.
The key functions that will usually be used by other classes are 
train() and TrainResult.

This implements the high-level predict-act-vote-payout loop that is described
in the write-up and the presentation.
"""

@dataclass(frozen=True)
class TrainResult(Generic[A, S]):
    """
    Should be treated as constant.

    Essentially a struct for containing all of the
    information about a given run of train()

    TODO: does not contain information about post-episode payouts

    histories: List[List[HistoryItem[A, S]]]
        contains all of the information about the votes
        that the history item contains

        Outer-most list indexes different episodes
        Inner List contains each timestep in each episode
        Each HistoryItem contains all of the information about
        one predict-act-vote-payout loop

    balances: Dict[Agent[A, S], float]
        Amount of money each agent is left with after train is finished
        being called. Not directly derivable because of post-episode
        payouts
    """
    histories: List[List[HistoryItem[A, S]]]
    balances: Dict[Agent[A, S], float]

    def history_item_for(self, 
            episode_num: int,
            t_step: int) -> HistoryItem[A, S]:
        """
        Essentially exists to make parsing TrainResult histories
        more clear

        Parameters
        ----------
        episode_num: int >= 0
            the index of the episode containing the returned
            history item
        t_step: int >= 0
            the timestep of the returned history item within the selected
            episode

        Returns
        -------
        history: HistoryItem[A, S]
            the selected history item containing all information about
            that timestep's loop.
        """
        return self.histories[episode_num][t_step]


def train(
        agents: List[Agent[A, S]],
        env: Environment[A, S],
        episode_seeds: Iterable[int],
        config: SystemConfiguration[A, B, S],
        tsteps_per_episode: int = np.inf) \
        -> TrainResult[A, S]:
    """

    Parameters
    ----------
    agents: List[Agent[A, S]]
        all of the agents that will have the opportunity to
        vote, predict & earn money during the timestep
    env: Environment[A, S]
        The environment that the agents are acting in
        All agents collectively take a single action each timestep.
        They all view the same resulting state (but may have different
        preferences about the quality of each state)
    episode_seeds: Iterable[int]
        Calls env.seed(seed) at the beginning of each episode.
        Each seed in episode_seeds is run once. Algorithm returns after
        no more seeds are left.
    config: SystemConfiguration[A, B, S]
        Configuration for thre training script, contains most of the
        algorithmic choices. See class documentation for more info
    tsteps_per_episode: int >= 0
        Runs each episode until either episode.done() is true or r
        tsteps_per_episode is exceeded
    Returns
    -------
    result: TrainResult[A, S]
        datatype logging the history of events during training
    """
    old_episode_history: List[List[HistoryItem[A, S]]] = []
    # history for previous episodes
    current_history: List[HistoryItem[A, S]] = []
    # history for current episode
    balances: Dict[Agent[A, S], float] = \
        {agent: 1. for agent in agents}
    # all agents start with $1
    config.voting_manager.set_n_agents(len(agents))
    # set_n_agents useful for checking max possible vote
    # and therefore max possible prediction

    seed: int
    for seed in episode_seeds:
        env.reset()
        env.seed(seed)
        # restart the environment each episode

        if len(current_history) > 0:
            old_episode_history.append(current_history)
            # note: this essentially obliviates episodes of 0 length
        current_history = []
        # clear current history

        t: int = 0
        while not env.done() and t < tsteps_per_episode:
            state: S = env.state()

            welfare_score: float = get_agent_votes(
                agents=agents,
                state=state,
                config=config)
            # aggregates agent votes about environment

            payouts: Dict[Agent[A, S], float] = calculate_payouts(
                current_history,
                welfare_score,
                config, t)

            agent: Agent[A, S]
            for agent in payouts:
                balances[agent] += payouts[agent]
            # give agents money proportional to their current payouts

            placed_bets: Dict[A, List[WeightedBet[A, S]]] = \
                get_agent_bets(agents, balances, env.state(), env.actions())
            # agents make predictions about the quality of each action

            action: A = select_action(
                placed_bets,
                config)
            # action selected based on predictions

            # TODO: make this its own function?
            bets_that_happened: List[WeightedBet[A, S]] = placed_bets[action]
            bet: WeightedBet[A, S]
            for bet in bets_that_happened:
                # duplicate assert
                assert sum(bet.bet) <= 1
                balances[bet.cast_by] *= (1 - sum(bet.bet))
            # only take money out of agent accounts for bets that actually happened
            # essentially 'refunds' bets on any actions that were not selected

            env.step(action)

            # update agent history
            ahi = AnonymizedHistoryItem()
            for agent in agents:
                agent.view(ahi)

            current_history.append(
                HistoryItem(
                    selected_action=action,
                    predictions=placed_bets,
                    t_enacted=t))
            # log the current timestep in history
            # used for returning results & calculating payouts
            t += 1
        
        final_payouts: Dict[Agent[A, S], float] = pay_outstanding_bets(
            current_history,t, config)

        for agent in final_payouts:
            balances[agent] += final_payouts[agent]
        # TODO: make receiving money a function?
        # without final_payouts, agents lose all money on any outstanding
        # bets when the episode ends
        # adding this ensures that the money in the system stays constant

    old_episode_history.append(current_history)
    # need to append the last one (TODO: redo this logic to avoid duplication)

    return TrainResult(
        histories=old_episode_history, 
        balances=balances)

def pay_outstanding_bets(
        history: List[HistoryItem[A, S]],
        last_t: int,
        config: SystemConfiguration[A, B, S]) -> Dict[Agent[A, S], float]:
    """
    Essentially pays out any money that agents still had left outstanding
    in bets when the episode ended (i.e. bets that hadn't had the chance
    to be paid back)

    All payouts are handled as if they were a normal timestep where the total
    votes were 0 (because nobody received any reward)

    TODO: configure payout scheme? (what if end of episode pays out average?)
    TODO: refactor this

    Parameters
    ----------
    history: List[HistoryItem[A, S]]
        record of all of the events that happened this episode
    last_t: int >= 0
        the last timestep before the episode ended
    config: SystemConfiguration[A, B, S]
        configuration used for payouts

    Returns
    -------
    payouts: Dict[Agent[A, S], float]
        the amount of money each agent is paid out for the outstanding
        bets
    """
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
    """
    Selects which action to collectively take for this timestep
    based on the predictions

    Parameters
    ----------
    bets: Dict[A, List[WeightedBet[A, S]]]
        for each action, the bets that were cast about the expected
        vote totals if that action is enacted
    config: SystemConfiguration[A, B, S]
        used for aggregating the bets & making decisions based on those
        bets
        really just uses config.policy_manager
    Returns
    -------
    action: A
        the selected action
    """
    # essentially just aggregates the bets & makes a decision based on the
    # aggregations
    aggregation: Dict[A, B] = config.policy_manager.aggregate_bets(bets)
    return config.policy_manager.select_action(aggregation)



def calculate_payouts(
        history: List[HistoryItem[A, S]],
        welfare_score: float,
        config: SystemConfiguration[A, B, S],
        t: int) \
        -> Dict[Agent[A, S], float]:
    """
    Calculates how much money each agent should receive at this
    timestep

    Parameters
    ----------
    history: List[HistoryItem[A, S]]
        All of the previous bets that have taken place
        this episode. Payouts are based on previous bets
    welfare_score: float >= 0
        the aggregated votes that all agents cast on this timestep.
        Should compare the predictions against this
    config: SystemConfiguration[A, B, S]
        uses config.payout_manager to calculate payouts
        uses all of config to validate prediction/bet validitiy
    t: int
        the timestep that payouts are being calculated for

        TODO: make sure that we don't calculate payouts for the same
        timestep twice?

    Returns
    -------
    total_payouts: Dict[Agent[A, S], float]
        the amount of money received by each agent
        all amounts should be positive
        not all agents necessarily receive payouts
        (i.e. if they bet 0 or had 0 money)
    """
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
    """
    Has each agent vote on the current state & aggregates their
    votes in to a single anonymized value

    Parameters
    ----------
    agents: List[Agent[A, S]]
        the agents that are eligible to vote
    state: S
        the current state that agents will base their vote upon
    config: SystemConfiguration[A, B, S]
        config.voting_manager is used to validate all votes
        and aggregate them

    Returns
    -------
    vote_total: float >= 0
        aggregated total votes received for the current timestep
        higher is better
    """
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
    """
    Solicits agent bets for each possible action available at this timestep
    Agents can opt out of betting by betting $0

    Parameters
    ----------
    agents: List[Agent[A, S]]
        the agents who may place bets at the current timestep
        should always be the same as the voters
    balances: Dict[Agent[A, S], float]
        the current amount of money that each agent has
        each agent can view their current account balance
        when making a decision
    state: S
        the current state of the environment, also used in agent
        decisions
    actions: List[A]
        the available actions at this timestep
    Returns
    -------
    placed_bets: Dict[A, List[WeightedBets[A, S]]]
        for each action, the list of bets that were placed on that action
        at most one bet per agent
        # TODO: technically multiple bets per agent could be useful
        # as a variance reduction strategy for agents
    """
    placed_bets: Dict[A, List[WeightedBet[A, S]]] = {}
    for action in actions:
        placed_bets[action] = []

        for agent in agents:
            money: float = balances[agent]
            bet: ActionBet = agent.bet(state, action, money)
            # solicit the agent bet

            wbet: WeightedBet[A, S] = WeightedBet(
                bet=bet.bet,
                prediction=bet.prediction,
                action=action,
                money=money,
                cast_by=agent)
            # attaches metadata about the bet to create a
            # WeightedBet
            placed_bets[action].append(wbet)

    return placed_bets
