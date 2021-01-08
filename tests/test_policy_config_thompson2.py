# -*- coding: utf-8 -*-
"""
file: test_policy_config_thompson.py

@author: Suhail.Alnahari

@description: This file tests thompson policy

@created: 2020-12-06T20:08:39.389Z-06:00

@last-modified: 2020-12-19T17:15:18.303Z-06:00
"""

# standard library
from typing import List, Dict

# 3rd party packages
import pytest
import numpy as np

# local source
from tests.conftest import floatIsEqual
import VIAYN.project_types as P
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import VIAYN.utils as U
from VIAYN.DiscreteDistribution import DiscreteDistribution


@pytest.mark.parametrize("bets,predictions,moneys,expectedVals,expectedProbs", [
    # single timestep random config start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [2,4,5],
        [0,1/1.4,0.4/1.4]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [2,2,2],
        [4,2,5],
        [1/1.4,0,0.4/1.4]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [2,4,5],
        [0.4/1.4,0.6/1.4,0.4/1.4]
    ),
    # single timestep random config end
    # multiple timestep random config start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,2,2],
        [4,8,10],
        [0.4/1.4,0.6/1.4,0.4/1.4],
    ),
    (
        [[0.2, 0.2], [0.3, 0.3], [0.2, 0.2]],
        [[2, 5], [4, 2], [5, 4]],
        [2,2,2],
        [7,6,9],
        [0.4/1.4,0.6/1.4,0.4/1.4],
    ),
    # multiple timestep random config end
    # only one agent bets, and not on every timestep start
    (
        [[0,0.0,0.0,0.0]],
        [[2,5,4,5]],
        [2],
        [-np.inf],
        [1]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet, and not on every timestep start
        (
        [[0,0.0,0.0,0.0],[0.1,0.1,0.1,0.1]],
        [[2,5,4,5],[4,2,8,3]],
        [2,5],
        [16,17],
        [0.,1.]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet in every timestep start
        (
        [[0.2,0.2,0.2],[0.3,0.3,0.3]],
        [[5,4,5],[2,8,3]],
        [2,5],
        [14,13],
        [0.4/1.9,1.5/1.9]
    ),
    # only one agent bets in every timestep end
])
def test_suggested_policy_config_single_action(
    bets,predictions,
    moneys,expectedVals,expectedProbs,
    gen_policy_conf,gen_weighted_bet):
    """
    This test checks that aggregate bets in policy config
    satsifies our definition of suggested aggregation 
    for at most a 3 agent system

    [bets] is the bet percentage of each agent in a certain timestep
    [predictions] is the corresponding prediction for each bet
    [moneys] is a list of the agents money at that time-step,
    [expectedVals] are the probablity keys of the object returned by aggregate bets
    [expectedProbs] are the probabilties of each key of the object returned by aggregate bets
    The other two parameters are test fixtures to help create
    objects easier.
    """

    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested)
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {'a':weightedBets}
    res = policyConf.aggregate_bets(dic)['a']
    assert(len(res.values) == len(expectedVals))
    assert(len(res.probabilities) == len(expectedProbs))
    for k in range(len(res.probabilities)):
        assert(floatIsEqual(res.values[k], expectedVals[k]))
        assert(floatIsEqual(res.probabilities[k], expectedProbs[k]))
    

@pytest.mark.parametrize("bets,predictions,moneys,expectedVals,expectedProbs", [
    # single timestep random config start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [2,4,5],
        [0,1/1.4,0.4/1.4]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [2,2,2],
        [4,2,5],
        [1/1.4,0,0.4/1.4]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [2,4,5],
        [0.4/1.4,0.6/1.4,0.4/1.4]
    ),
    # single timestep random config end
    # multiple timestep random config start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,2,2],
        [4,8,10],
        [0.4/1.4,0.6/1.4,0.4/1.4],
    ),
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,5],[4,2],[5,4]],
        [2,2,2],
        [7,6,9],
        [0.4/1.4,0.6/1.4,0.4/1.4],
    ),
    # multiple timestep random config end
    # only one agent bets, and not on every timestep start
    (
        [[0,0.0,0.0,0.0]],
        [[2,5,4,5]],
        [2],
        [-np.inf],
        [1]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet, and not on every timestep start
        (
        [[0,0.,0.,0.],[0.2,0.2,0.2,0.2]],
        [[2,5,4,5],[4,2,8,3]],
        [2,5],
        [16,17],
        [0.,1.]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet in every timestep start
        (
        [[0.2,0.2,0.2],[0.3,0.3,0.3]],
        [[5,4,5],[2,8,3]],
        [2,5],
        [14,13],
        [0.4/1.9,1.5/1.9]
    ),
    # only one agent bets in every timestep end
])
def test_suggested_policy_config_multiple_actions(
    bets,predictions,
    moneys,expectedVals,expectedProbs,
    gen_policy_conf,gen_weighted_bet):
    """
    This test checks that aggregate bets in policy config
    satsifies our definition of suggested aggregation 
    for at most a 3 agent system where the agents have multiple IDENTICAL actions

    [bets] is the bet percentage of each agent in a certain timestep
    [predictions] is the corresponding prediction for each bet
    [moneys] is a list of the agents money at that time-step,
    [expectedVals] are the probablity keys of the object returned by aggregate bets
    [expectedProbs] are the probabilties of each key of the object returned by aggregate bets
    The other two parameters are test fixtures to help create
    objects easier.
    """

    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested)
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {i:weightedBets for i in range(100)}
    for i in range(len(dic)):
        res = policyConf.aggregate_bets(dic)[i]
        assert(len(res.values) == len(expectedVals))
        assert(len(res.probabilities) == len(expectedProbs))
        for k in range(len(res.probabilities)):
            assert(floatIsEqual(res.values[k], expectedVals[k]))
            assert(floatIsEqual(res.probabilities[k], expectedProbs[k]))


class constantDistribution(DiscreteDistribution):
    def __init__(self,constant:float):
        self.constant: float = constant
    def sample(self) -> float:
        return self.constant



@pytest.mark.parametrize("arr,vals,expected", [
    # random configurations with 6 actions start
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5),
    ([5,4,3,2,1,0],[0,0,0,0,0,100],0),
    ([0,1,2,3,4,5],[0,10,100000,1000,10000,100],2),
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],5),
    # random configurations with 6 actions end
    # more distributions than actions
    ([0,1,2],[0,0,3],2), # TODO: should fail
    # more distributions than actions
    # slight difference between actions start
    ([1,2],[1,0.999],1),
    ([2,1],[1,0.999],2),
    # slight difference between actions end
])
def test_suggested_policy_config_select_action(arr,vals,expected,gen_policy_conf):
    """
    This test checks that select action in policy config
    satsifies our definition of suggested select action

    [arr] is the list of actions available
    [vals] is the list of values for each constant distribution [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested)
    aggregate_bets : Dict[float,List[DiscreteDistribution]] = {}
    for k in range(len(arr)):
        aggregate_bets[arr[k]]= constantDistribution(vals[k])
    # test is ran 100 times to ensure consistency 
    # (constant distrib guarantees that to some extent)
    for _ in range(100):
        res = policyConf.select_action(aggregate_bets)
        assert(res == expected)

@pytest.mark.parametrize("arr,vals,expected", [
    # random configurations with 6 actions start
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5),
    ([5,4,3,2,1,0],[0,0,0,0,0,100],0),
    ([0,1,2,3,4,5],[0,10,100000,1000,10000,100],2),
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],5),
    # random configurations with 6 actions end
    # more distributions than actions
    ([0,1,2],[0,0,3],2),
    # more distributions than actions
    # slight difference between actions start
    ([1,2],[1,0.999],1),
    ([2,1],[1,0.999],2),
    # slight difference between actions end
])
def test_suggested_policy_config_action_probs(arr,vals,expected,gen_policy_conf):
    """
    This test checks that action probs in policy config
    satsifies our definition of suggested action probabilities which is identical
    to select action but gives out probabilities.

    [arr] is the list of actions available
    [vals] is the list of values for each action in [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested)
    aggregate_bets : Dict[float,List[DiscreteDistribution]] = {}
    for k in range(len(arr)):
        aggregate_bets[arr[k]]= constantDistribution(vals[k])
    if expected is not None:
        res = policyConf.action_probabilities(aggregate_bets)
        for i in arr:
            if (i == expected):
                assert(floatIsEqual(res[i],1))
                continue
            assert(floatIsEqual(res[i],0))
    else:
        with pytest.raises(Exception):
            policyConf.action_probabilities(aggregate_bets)


class OneGoodOneBadDistribution:
    def __init__(self, constant: float):
        self.constant: float = constant
        self.good: bool = False
    def sample(self) -> float:
        self.good: bool = not(self.good)
        if (self.good):
            return self.constant
        return -1*self.constant

@pytest.mark.parametrize("arr,vals,expected", [
    # more distributions for some actions start
    ([0,1,2],[0,0,0],0.333),
    # more distributions for some actions end
    # Equal reward start
    ([1,2],[100,100],1./2),
    ([2,1],[0,0],1./2),
    # Equal reward end
    # slight difference between actions start
    ([1,2],[1,0.999],1./2),
    # slight difference between actions end
])
def test_suggested_policy_config_action_probs_uniform(arr,vals,expected,gen_policy_conf):
    """
    This test checks that action probs in policy config
    satsifies our definition of suggested action probabilities which is identical
    to select action but gives out probabilities.

    [arr] is the list of actions available
    [vals] is the list of values for each action in [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested)
    aggregate_bets : Dict[float,List[DiscreteDistribution]] = {}
    for k in range(len(arr)):
        aggregate_bets[arr[k]]= OneGoodOneBadDistribution(vals[k])
    res = policyConf.action_probabilities(aggregate_bets)
    for i in arr:
        assert floatIsEqual(res[i],expected,0.05) # NOTE: this is a pretty big tolerance