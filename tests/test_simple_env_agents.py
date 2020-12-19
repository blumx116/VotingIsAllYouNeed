# -*- coding: utf-8 -*-
"""
Created on 2020-12-03 19:25:18
@author: suhail

This file tests simple agents in a simple environment
"""

# standard library
import random

# 3rd party packages
import pytest
import numpy as np

# local source
import VIAYN.project_types as project_types
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
from tests.conftest import floatIsEqual


def test_constant_simple_agent_basic(constant_agent_config):
    """
    This test checks that a constant agent predicts and votes the configured
    [AB] and [vote] for different states in 100 timesteps.

    [AB] is a specified ActionBet
    [vote] is a specified float
    """
    # for a list of random constant agent configurations:
    for AB,vote in constant_agent_config:
        print(f"constant: {AB.bet} and {AB.prediction}, {vote}")
        agent = fac.AgentFactory.create(
            fac.AgentFactorySpec(
                fac.AgentsEnum.constant,
                vote,
                bet=AB.bet,
                prediction=AB.prediction
            )
        )
        env = fac.EnvFactory.create(
            fac.EnvsFactorySpec(
                fac.EnvsEnum.default,
                n_actions=2
            )
        )
        for _ in range(100):
            state = env.state()
            assert(agent.vote(state) == vote)
            for j in env.actions():
                action_bet = agent.bet(state,j,1)
                assert(
                    len(action_bet.bet) == len(AB.bet)
                )
                assert(
                    len(action_bet.prediction) == len(AB.prediction)
                )
                for k in range(len(action_bet.bet)):
                    assert(floatIsEqual(action_bet.bet[k], AB.bet[k]))
                for k in range(len(action_bet.prediction)):
                    assert(floatIsEqual(action_bet.prediction[k], AB.prediction[k]))
            env.step(random.choice(env.actions()))

def test_random_simple_agent_basic(random_agent_config):
    """
    This test checks that a random agent predicts and votes using
    the configured [seed] and [vote] for different states in 
    100 timesteps.

    [seed] is a specified seed for bets and predictions where they both
        have different generators
    [vote] is a specified float
    """
    for vote, seed in random_agent_config:
        print(f"random: {vote} , {seed}")
        votingConf = fac.VotingConfigFactory.create(
            fac.VotingConfigFactorySpec(fac.VotingConfigEnum.simple,vote_range.BinaryVoteRange)
        )
        agent = fac.AgentFactory.create(
            fac.AgentFactorySpec(
                fac.AgentsEnum.random,
                vote,
                bet=0.5,
                N=1,
                seed=seed,
                totalVotesBound=(
                    votingConf.min_possible_vote_total,
                    votingConf.max_possible_vote_total
                )
            )
        )
        env = fac.EnvFactory.create(
            fac.EnvsFactorySpec(
                fac.EnvsEnum.default,
                n_actions=2
            )
        )
        genPred = np.random.default_rng(seed=seed)
        for _ in range(100):
            state = env.state()
            assert(agent.vote(state) == vote)
            for j in env.actions():
                action_bet = agent.bet(state, j, 1)
                for k in range(len(action_bet.bet)):
                    assert(floatIsEqual(action_bet.bet[k], 0.5))
                for k in range(len(action_bet.prediction)):
                    min: float = votingConf.min_possible_vote_total()
                    max: float = votingConf.max_possible_vote_total()
                    if not np.isfinite(min):
                        min = -100.
                    if not np.isfinite(max):
                        max = 100
                    assert(
                        floatIsEqual(
                            action_bet.prediction[k], 
                            genPred.uniform(min, max)
                        )
                    )
            env.step(random.choice(env.actions()))


@pytest.mark.parametrize("N", [
    1,
    2,
    5,
    20,
    15
])
def test_random_simple_agent_forward_prediction(N):
    """
    This test checks that a random agent predicts [N] timesteps
    using the configured seed = 0 and vote = 5 for different states in 
    100 timesteps.

    [N] is the length of the time-steps
    """
    vote, seed = (5,0)
    print(f"random: {vote} , {seed}")
    votingConf = fac.VotingConfigFactory.create(
        fac.VotingConfigFactorySpec(fac.VotingConfigEnum.simple,vote_range.BinaryVoteRange)
    )
    agent = fac.AgentFactory.create(
        fac.AgentFactorySpec(
            fac.AgentsEnum.random,
            vote,
            bet=0.5,
            N=N,
            seed=seed,
            totalVotesBound=(
                votingConf.min_possible_vote_total,
                votingConf.max_possible_vote_total
            )
        )
    )
    env = fac.EnvFactory.create(
        fac.EnvsFactorySpec(
            fac.EnvsEnum.default,
            n_actions=5
        )
    )    
    genBet = np.random.default_rng(seed=seed)
    genPred = np.random.default_rng(seed=seed)
    for _ in range(100):
        state = env.state()
        assert(agent.vote(state) == vote)
        for j in env.actions():
            action_bet = agent.bet(state,j,1)
            assert(
                len(action_bet.bet) == N
            )
            assert(
                len(action_bet.prediction) == N
            )
            for k in range(len(action_bet.bet)):
                assert(floatIsEqual(action_bet.bet[k], 0.5))
            for k in range(len(action_bet.prediction)):
                min: float = votingConf.min_possible_vote_total()
                max: float = votingConf.max_possible_vote_total()
                if not np.isfinite(min):
                    min = -100.
                if not np.isfinite(max):
                    max = 100
                assert(
                    floatIsEqual(
                        action_bet.prediction[k], 
                        genPred.uniform(
                            min, max)
                    )
                )


############ NOT ALLOWED ############
# @pytest.mark.parametrize("config",#
#     random_agent_config           #
# )                                 #
# def test_checking(config):        #
#     print(config)                 #
#####################################
