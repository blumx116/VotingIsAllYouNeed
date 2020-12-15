# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:18:03 2020
@author: suhail
"""

import pytest
from VIAYN.project_types import (
    ActionBet, A, WeightedBet, Agent, HistoryItem,
    VoteBoundGetter, VoteRange
)
import VIAYN.samples.factory as factory
import VIAYN.samples.vote_ranges as vote_range
from typing import (
    Optional, Generic, TypeVar, List,
    Iterable, Dict, Callable, Tuple,
    Union
) 
import numpy as np

@pytest.fixture
def constant_agent_config():
    return [
    (ActionBet([0.],[5.]) , 5.),
    (ActionBet([0.],[4.]) , 5.),
    (ActionBet([0.5],[1.]) , 5.),
    (ActionBet([0.5],[0.]), 0.),
    (ActionBet([0.5],[2.]), 2.),
    (ActionBet([0.5],[2.]), 1.),
    (ActionBet([0.5],[7.]), 10.),
    (ActionBet([0.,0.43,0.1215,0.8,1.],[4.,1,4,1,4]), 9.),
    (ActionBet([0.,0.1215,0.8,1.],[4.,4,1,4]), 1.),
    (ActionBet([0.,0.43,0.1215,0.8],[4.,1,4,1]), 2.),
    (ActionBet([0.,0.43,0.8,1.],[4.,1,4,4]), 7.),
    (ActionBet([0.1215,0.8,1.],[1,1,4]), 5.),
    (ActionBet([0.,1.],[1,4]), 9.5)
]

@pytest.fixture
def random_agent_config():
    return  [
    (5.,0),
    (5., 123),    
    (0., 1231131),
    (1., 1235234),
    (2., 5124),
    (8., 82384),
    (10., 0),
]

def floatIsEqual(num1: float,num2: float) -> bool:
    return abs(num1-num2) < 0.000001

@pytest.fixture
def gen_agent():
    def _gen_agent_(
        agentType: factory.AgentsEnum,
        vote: float,
        totalVotesBound: Optional[Tuple[VoteBoundGetter, VoteBoundGetter]] = None,
        seed: Optional[int] = None,
        prediction: Optional[Union[float, List[float]]] = None,
        bet: Optional[Union[float, List[float]]] = None,
        N: Optional[int] = None,
        ):
        return factory.AgentFactory.create(
            factory.AgentFactorySpec(
                agentType,
                vote,
                totalVotesBound,
                seed,
                prediction,
                bet,
                N
            )
        )
    return _gen_agent_

@pytest.fixture
def gen_env():
    def _gen_env_(envType: factory.EnvsEnum):
        return factory.EnvFactory.create(
            factory.EnvsFactorySpec(envType)
        )
    return _gen_env_

@pytest.fixture
def gen_vote_conf():
    def _gen_vote_conf_(
        configType: factory.VotingConfigEnum,
        voteRange: VoteRange,
        n_agents: Optional[int] = 0
    ):
        return factory.VotingConfigFactory.create(
            factory.VotingConfigFactorySpec(configType, voteRange,n_agents)
        )
    return _gen_vote_conf_

@pytest.fixture
def gen_policy_conf():
    def _gen_policy_conf_(configType: factory.PolicyConfigEnum):
        return factory.PolicyConfigFactory.create(
            factory.PolicyConfigFactorySpec(configType)
        )
    return _gen_policy_conf_

@pytest.fixture
def gen_payout_conf():
    def _gen_payout_conf_(configType: factory.PayoutConfigEnum):
        return factory.PayoutConfigFactory.create(
            factory.PayoutConfigFactorySpec(configType)    
        )
    return _gen_payout_conf_

@pytest.fixture
def gen_weighted_bet():
    def _gen_weighted_bet_(
        bet: List[float],
        prediction: List[float], 
        action: A = None,
        money: float =0,
        cast_by = None):
        return WeightedBet(bet,prediction,action,money,cast_by)
    return _gen_weighted_bet_

@pytest.fixture
def gen_history_item():
    def _gen_history_item_(
        selectedA,
        predictions = {}, 
        t_enacted: int = 0
        ):
        return HistoryItem(selectedA,predictions,t_enacted)
    return _gen_history_item_

    