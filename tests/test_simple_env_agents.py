# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 19:25:18
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 15:02:25

import pytest
import VIAYN.project_types as project_types
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import numpy as np

from tests.conftest import floatIsEqual


def test_constant_simple_agent_basic(constant_agent_config):
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
    

def test_random_simple_agent_basic(random_agent_config):
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


@pytest.mark.parametrize("N", [
    1,
    2,
    5,
    20,
    15
])
def test_random_simple_agent_forward_prediction(N):
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
