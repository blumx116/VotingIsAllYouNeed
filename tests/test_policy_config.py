# -*- coding: utf-8 -*-
"""
Created on 2020-12-06 18:09:26
@author: suhail

This file tests policy configurations
"""

# standard library
from typing import List

# 3rd party packages
import pytest

# local source
from tests.conftest import floatIsEqual
import VIAYN.project_types as P
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import numpy as np
import VIAYN.utils as U



@pytest.mark.parametrize("bets,predictions,moneys,expected", [
    # varying bets and predictions for a three agent system over one time-step start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [0+(4./1.4)+(2./1.4)]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [2,2,2],
        [(4./1.4)+0+(2./1.4)]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [(0.8/1.4)+((0.6*4)/1.4)+(2./1.4)]
    ),
    # varying bets and predictions for a three agent system over one time-step end
    # varying bets and predictions for a three agent system over two time-steps start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,2,2],
        [(0.8/1.4)+((0.6*4)/1.4)+(2./1.4), (0.8/1.4)+((0.6*4)/1.4)+(2./1.4)]
    ),
    (
        [[0.2,0.2],[0.3,0.2],[0.2,0.3]],
        [[2,5],[4,2],[5,4]],
        [2,2,2],
        [(0.8/1.4)+((0.6*4)/1.4)+(2./1.4), (0.8/1.4)+((0.6*4)/1.4)+(2./1.4)]
    ),
    # varying bets and predictions for a three agent system over two time-steps end
    # varying money for a three agent system over one time-step start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,4],
        [0+(4./1.8)+(4./1.8)]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [4,2,2],
        [(8./2.4)+0+(2./2.4)]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    ),
    # varying money for a three agent system over one time-step end
    # varying money for a three agent system over two time-steps start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1), (0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    )
    # varying money for a three agent system over two time-steps end
])
def test_simple_policy_config_single_action(
    bets,predictions,
    moneys,expected,
    gen_policy_conf,gen_weighted_bet):
    """
    This test checks that aggregate bets in policy config
    satsifies our definition os simple aggregation for a 3 agent
    system

    [bets] is the bet percentage of each agent in a certain timestep
    [predictions] is the corresponding prediction for each bet
    [moneys] is a list of the agents money at that time-step
    The other two parameters are test fixtures to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.simple)
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {'a':weightedBets}
    res = policyConf.aggregate_bets(dic)['a']
    assert(floatIsEqual(res,sum(expected)))
    # adding sum was easier than changing expected to be a float

@pytest.mark.parametrize("bets,predictions,moneys,expected", [
    # varying bets and predictions for a three agent system over one time-step start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [0+(4./1.4)+(2./1.4)]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [2,2,2],
        [(4./1.4)+0+(2./1.4)]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [(0.8/1.4)+((0.6*4)/1.4)+(2./1.4)]
    ),
    # varying bets and predictions for a three agent system over one time-step end
    # varying bets and predictions for a three agent system over two time-steps start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,2,2],
        [(0.8/1.4)+((0.6*4)/1.4)+(2./1.4), (0.8/1.4)+((0.6*4)/1.4)+(2./1.4)]
    ),
    (
        [[0.2,0.2],[0.3,0.2],[0.2,0.3]],
        [[2,5],[4,2],[5,4]],
        [2,2,2],
        [(0.8/1.4)+((0.6*4)/1.4)+(2./1.4), (0.8/1.4)+((0.6*4)/1.4)+(2./1.4)]
    ),
    # varying bets and predictions for a three agent system over two time-steps end
    # varying money for a three agent system over one time-step start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,4],
        [0+(4./1.8)+(4./1.8)]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [4,2,2],
        [(8./2.4)+0+(2./2.4)]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    ),
    # varying money for a three agent system over one time-step end
    # varying money for a three agent system over two time-steps start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1), (0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    )
    # varying money for a three agent system over two time-steps end
])
def test_simple_policy_config_multiple_actions(
    bets,predictions,
    moneys,expected,
    gen_policy_conf,gen_weighted_bet):
    """
    This test checks that aggregate bets in policy config
    satsifies our definition of simple aggregation for a 3 agent
    system where the agents have multiple IDENTICAL actions

    [bets] is the bet percentage of each agent in a certain timestep
    [predictions] is the corresponding prediction for each bet
    [moneys] is a list of the agents money at that time-step
    The other two parameters are test fixtures to help create
    objects easier.
    """
    
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.simple)
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {i:weightedBets for i in range(100)}
    res = policyConf.aggregate_bets(dic)
    for i in res.keys():
        assert(floatIsEqual(res[i], sum(expected)))


@pytest.mark.parametrize("arr,vals,expected", [
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5), # best action satisfies highest index
    ([5,4,3,2,1,0],[0,0,0,0,0,100],0), # best action is the lowest index
    ([0,1,2,3,4,5],[0,10,100,1000,10000,100000],5), # actions have varying values
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],5), # actions with varying values are not in order
    ([0,1,2,3,4,5],[0,10,100,10000000,100, 10],3), # best action is the in middle
    # ([1,0,2,3,4,5],[0,0,0,0,0,0],1), # TODO: all actions have the same value should be tested
    # ([],[],None), # TODO: should test that this throws
])
def test_simple_policy_config_select_action(arr,vals,expected,gen_policy_conf):
    """
    This test checks that select action in policy config
    satsifies our definition of simple select action

    [arr] is the list of actions available
    [vals] is the list of values for each action in [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.simple)
    res = policyConf.select_action({key:val for key,val in zip(arr,vals)})
    assert(res == expected)

@pytest.mark.parametrize("arr,vals,expected", [
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5), # best action satisfies highest index
    ([5,4,3,2,1,0],[0,0,0,0,0,100],0), # best action is the lowest index
    ([0,1,2,3,4,5],[0,10,100,1000,10000,100000],5), # actions have varying values
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],5), # actions with varying values are not in order
    ([0,1,2,3,4,5],[0,10,100,10000000,100, 10],3), # best action is the in middle
    # ([1,0,2,3,4,5],[0,0,0,0,0,0],1), # TODO: all actions have the same value should be tested
    # ([],[],None), # TODO: should test that this throws
])
def test_simple_policy_config_action_probs(arr,vals,expected,gen_policy_conf):
    """
    This test checks that action probs in policy config
    satsifies our definition of simple action probabilities which is identical
    to select action but gives out probabilities.

    [arr] is the list of actions available
    [vals] is the list of values for each action in [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.simple)
    res = policyConf.action_probabilities({key:val for key,val in zip(arr,vals)})
    try:
        for i in arr:
            if (i == expected):
                assert(floatIsEqual(res[i],1))
                continue
            assert(floatIsEqual(res[i],0))
    except:
        assert expected is None
        # TODO: this technically catches all exceptions, which is no bueno
