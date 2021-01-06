# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:18:03 2020
@author: suhail

This file stores pytest test fixtures, which are common
data structures and methods used in our tests.
"""


# standard library
from typing import (
    Optional, Generic, TypeVar, List,
    Iterable, Dict, Callable, Tuple,
    Union, Sequence
) 

# 3rd party packages
import pytest
import numpy as np

# local source
from VIAYN.project_types import (
    ActionBet, A, WeightedBet, Agent, HistoryItem,
    VoteBoundGetter, VoteRange
)
import VIAYN.samples.factory as factory
import VIAYN.samples.vote_ranges as vote_range


@pytest.fixture
def constant_agent_config():
    """
    Random configurations for constant agents
    the first parameter is what agent will use to bet
        the second is vote
    """
    return [
    (ActionBet([0.],[5.]) , 5.),
    (ActionBet([0.],[4.]) , 5.),
    (ActionBet([0.5],[0.25]) , 5.),
    (ActionBet([0.5],[0.]), 0.),
    (ActionBet([0.5],[0.499]), 2.),
    (ActionBet([0.0],[0.0]), 1.),
    (ActionBet([0.5],[0.2]), 10.),
    (ActionBet([0.,0.43,0.1215,0.1],[4.,1,4,1]), 9.),
    (ActionBet([0.,0.1215,0.5,0.2],[4.,4,1,4]), 1.),
    (ActionBet([0.,0.43,0.1215,0.0],[4.,1,4,1]), 2.),
    (ActionBet([0.,0.13,0.8,0.05],[4.,1,4,4]), 7.),
    (ActionBet([0.1215,0.5,0.2],[1,1,4]), 5.),
    (ActionBet([0.,1.],[1,4]), 9.5)
]

@pytest.fixture
def random_agent_config():
    """
    Random configurations for random agents
    the first parameter is vote the second is seed
    """
    return  [
    (5.,0),
    (5., 123),    
    (0., 1231131),
    (1., 1235234),
    (2., 5124),
    (8., 82384),
    (10., 0),
]

def floatIsEqual(num1: float,num2: float,epsilon: float = 1e-6) -> bool:
    return abs(num1-num2) < epsilon

def sequenceEqual(l1: Sequence[float], l2: Sequence[float], epsilon: float=1e-6) -> bool:
    """
    checks if all elements of a sequence of floats (or ints) are the same
    returns false if length is not equal
    NOTE: does not c heck shape
    Parameters
    ----------
    l1: Sequence[float]
        first list, tuple or np.ndarray to compare
    l2: Sequence[float]
        second list, tupl or np.ndarray to compare
    epsilon: float >= 0
        used for floatIsEqual

    Returns
    equal: bool
    -------

    """
    if not len(l1) == len(l2):
        return False
    for el1, el2 in zip(l1, l2):
        if not floatIsEqual(el1, el2, epsilon):
            return False
    return True



@pytest.fixture
def gen_agent():
    """
    Function to make creating agents easier by configuring everything for an agent
    and creating it in one call instead of needing to create multiple objects 
    every time to create an agent
    """

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
    """
    Function to make creating envs easier by configuring everything for an env
    and creating it in one call instead of needing to create multiple objects 
    every time to create an env
    """
    def _gen_env_(envType: factory.EnvsEnum):
        return factory.EnvFactory.create(
            factory.EnvsFactorySpec(envType)
        )
    return _gen_env_

@pytest.fixture
def gen_vote_conf():
    """
    Function to make creating a vote config easier by configuring everything for a vote config
    and creating it in one call instead of needing to create multiple objects 
    every time to create an vote config
    """
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
    """
    Function to make creating a policy config easier by configuring everything for a policy config
    and creating it in one call instead of needing to create multiple objects 
    every time to create an policy config
    """
    def _gen_policy_conf_(configType: factory.PolicyConfigEnum):
        return factory.PolicyConfigFactory.create(
            factory.PolicyConfigFactorySpec(configType)
        )
    return _gen_policy_conf_

@pytest.fixture
def gen_payout_conf():
    """
    Function to make creating a payout config easier by configuring everything for a payout config
    and creating it in one call instead of needing to create multiple objects 
    every time to create an payout config
    """
    def _gen_payout_conf_(configType: factory.PayoutConfigEnum):
        return factory.PayoutConfigFactory.create(
            factory.PayoutConfigFactorySpec(configType)    
        )
    return _gen_payout_conf_

@pytest.fixture 
def gen_weighted_bet():
    """
    Function to make creating a weighted bet easier by configuring only what we want to configure
    for specific tests instead of having to specify every parameter.
    """
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
    """
    Function to make creating a history item easier by configuring only what we want to configure
    for specific tests instead of having to specify every parameter.
    """
    def _gen_history_item_(
        selectedA,
        predictions = None, 
        t_enacted: int = 0):
        if predictions is None:
            predictions = {}
        return HistoryItem(selectedA,predictions,t_enacted)
    return _gen_history_item_

    