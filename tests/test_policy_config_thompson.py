# -*- coding: utf-8 -*-
"""
file: test_policy_config_thompson.py

@author: Suhail.Alnahari

@description: This file tests general thompson policy

@created: 2020-12-06T20:08:39.389Z-06:00

@last-modified: 2020-12-19T17:02:25.596Z-06:00
"""

# standard library
from typing import List

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
        [[2,4,5]],
        [[0,1/1.4,0.4/1.4]]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [2,2,2],
        [[4,2,5]],
        [[1/1.4,0,0.4/1.4]]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [[2,4,5]],
        [[0.4/1.4,0.6/1.4,0.4/1.4]]
    ),
    # single timestep random config end
    # multiple timestep random config start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,2,2],
        [[2,4,5],[2,4,5]],
        [
            [0.4/1.4,0.6/1.4,0.4/1.4],
            [0.4/1.4,0.6/1.4,0.4/1.4]
        ]
    ),
    (
        [[0.2,0.2],[0.3,0.2],[0.2,0.3]],
        [[2,5],[4,2],[5,4]],
        [2,2,2],
        [[2,4,5],[5,2,4]],
        [[0.4/1.4,0.6/1.4,0.4/1.4],[0.4/1.4,0.4/1.4,0.6/1.4]]
    ),
    # multiple timestep random config end
    # only one agent bets, and not on every timestep start
    (
        [[0,0.2,0.3,0.4]],
        [[2,5,4,5]],
        [2],
        [[2],[5],[4],[5]],
        [[1],[1],[1],[1]]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet, and not on every timestep start
        (
        [[0,0.2,0.3,0.4],[0,0.3,0.4,0.2]],
        [[2,5,4,5],[4,2,8,3]],
        [2,5],
        [
            [2,4],
            [5,2],
            [4,8],
            [5,3]
        ],
        [
            [0.5,0.5],
            [0.4/1.9,1.5/1.9],
            [0.6/2.6,2/2.6],
            [0.8/1.8,1/1.8]
        ]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet in every timestep start
        (
        [[0.2,0.3,0.4],[0.3,0.4,0.2]],
        [[5,4,5],[2,8,3]],
        [2,5],
        [
            [5,2],
            [4,8],
            [5,3]
        ],
        [
            [0.4/1.9,1.5/1.9],
            [0.6/2.6,2/2.6],
            [0.8/1.8,1/1.8]
        ]
    ),
    # only one agent bets in every timestep end
])
def test_suggested_policy_config_single_action(
    bets,predictions,
    moneys,expectedVals,expectedProbs,
    gen_policy_conf,gen_weighted_bet):
    """
    This test checks that aggregate bets in policy config
    satsifies our definition of suggested general aggregation 
    for at most a 3 agent system

    [bets] is the bet percentage of each agent in a certain timestep
    [predictions] is the corresponding prediction for each bet
    [moneys] is a list of the agents money at that time-step,
    [expectedVals] are the probablity keys of the object returned by aggregate bets
    [expectedProbs] are the probabilties of each key of the object returned by aggregate bets
    The other two parameters are test fixtures to help create
    objects easier.
    """

    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {'a':weightedBets}
    res = policyConf.aggregate_bets(dic)['a']
    assert(len(expectedVals) == len(res))
    for t in range(len(res)):
        assert(len(res[t].values) == len(expectedVals[t]))
        assert(len(res[t].probabilities) == len(expectedProbs[t]))
        for k in range(len(res[t].probabilities)):
            assert(floatIsEqual(res[t].values[k], expectedVals[t][k]))
            assert(floatIsEqual(res[t].probabilities[k], expectedProbs[t][k]))
    

@pytest.mark.parametrize("bets,predictions,moneys,expectedVals,expectedProbs", [
    # single timestep random config start
    (
        [[0],[0.5],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [[2,4,5]],
        [[0,1/1.4,0.4/1.4]]
    ),
    (
        [[0.5],[0],[0.2]],
        [[4],[2],[5]],
        [2,2,2],
        [[4,2,5]],
        [[1/1.4,0,0.4/1.4]]
    ),
    (
        [[0.2],[0.3],[0.2]],
        [[2],[4],[5]],
        [2,2,2],
        [[2,4,5]],
        [[0.4/1.4,0.6/1.4,0.4/1.4]]
    ),
    # single timestep random config end
    # multiple timestep random config start
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,2,2],
        [[2,4,5],[2,4,5]],
        [
            [0.4/1.4,0.6/1.4,0.4/1.4],
            [0.4/1.4,0.6/1.4,0.4/1.4]
        ]
    ),
    (
        [[0.2,0.2],[0.3,0.2],[0.2,0.3]],
        [[2,5],[4,2],[5,4]],
        [2,2,2],
        [[2,4,5],[5,2,4]],
        [[0.4/1.4,0.6/1.4,0.4/1.4],[0.4/1.4,0.4/1.4,0.6/1.4]]
    ),
    # multiple timestep random config end
    # only one agent bets, and not on every timestep start
    (
        [[0,0.2,0.3,0.4]],
        [[2,5,4,5]],
        [2],
        [[2],[5],[4],[5]],
        [[1],[1],[1],[1]]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet, and not on every timestep start
        (
        [[0,0.2,0.3,0.4],[0,0.3,0.4,0.2]],
        [[2,5,4,5],[4,2,8,3]],
        [2,5],
        [
            [2,4],
            [5,2],
            [4,8],
            [5,3]
        ],
        [
            [0.5,0.5],
            [0.4/1.9,1.5/1.9],
            [0.6/2.6,2/2.6],
            [0.8/1.8,1/1.8]
        ]
    ),
    # only one agent bets, and not on every timestep end
    # multiple agents bet in every timestep start
        (
        [[0.2,0.3,0.4],[0.3,0.4,0.2]],
        [[5,4,5],[2,8,3]],
        [2,5],
        [
            [5,2],
            [4,8],
            [5,3]
        ],
        [
            [0.4/1.9,1.5/1.9],
            [0.6/2.6,2/2.6],
            [0.8/1.8,1/1.8]
        ]
    ),
    # only one agent bets in every timestep end
])
def test_suggested_policy_config_multiple_actions(
    bets,predictions,
    moneys,expectedVals,expectedProbs,
    gen_policy_conf,gen_weighted_bet):
    """
    This test checks that aggregate bets in policy config
    satsifies our definition of suggested general aggregation 
    for at most a 3 agent system where the agents have multiple IDENTICAL actions

    [bets] is the bet percentage of each agent in a certain timestep
    [predictions] is the corresponding prediction for each bet
    [moneys] is a list of the agents money at that time-step,
    [expectedVals] are the probablity keys of the object returned by aggregate bets
    [expectedProbs] are the probabilties of each key of the object returned by aggregate bets
    The other two parameters are test fixtures to help create
    objects easier.
    """

    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {i:weightedBets for i in range(100)}
    for i in range(len(dic)):
        res = policyConf.aggregate_bets(dic)[i]
        assert(len(expectedVals) == len(res))
        for t in range(len(res)):
            assert(len(res[t].values) == len(expectedVals[t]))
            assert(len(res[t].probabilities) == len(expectedProbs[t]))
            for k in range(len(res[t].probabilities)):
                assert(floatIsEqual(res[t].values[k], expectedVals[t][k]))
                assert(floatIsEqual(res[t].probabilities[k], expectedProbs[t][k]))

class constantDistribution(DiscreteDistribution):
    def __init__(self,constant:float):
        self.constant: float = constant
    def sample(self) -> float:
        return self.constant



@pytest.mark.parametrize("arr,vals,expected", [
    # random configurations with 6 actions start
    ([0,1,2,3,4,5],[[0],[0],[0],[0],[0],[100]],5),
    ([5,4,3,2,1,0],[[0],[0],[0],[0],[0],[100]],0),
    ([0,1,2,3,4,5],[[0],[10],[100000],[1000],[10000],[100]],2),
    ([5,4,3,2,1,0],[[100000],[10000],[1000],[100],[10],[0]],5),
    # random configurations with 6 actions end
    # more distributions for some actions start
    ([0,1,2],[[0],[0,1,2,3],[0]],None), # TODO: should fail
    # more distributions for some actions end
    # high reward later in the future start
    ([1,2],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,100000]],2),
    ([2,1],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,100000]],1),
    # high reward later in the future end
    # slight difference between actions start
    ([1,2],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,0.999]],1),
    ([2,1],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,0.999]],2),
    # slight difference between actions end
])
def test_suggested_policy_config_select_action(arr,vals,expected,gen_policy_conf):
    """
    This test checks that select action in policy config
    satsifies our definition of suggested general select action

    [arr] is the list of actions available
    [vals] is the list of list values for each constant distribution [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    aggregate_bets : Dict[float,List[DiscreteDistribution]] = {}
    for k in range(len(arr)):
        aggregate_bets[arr[k]]= [constantDistribution(i) for i in vals[k]]
    # test is ran 100 times to ensure consistency 
    # (constant distrib guarantees that to some extent)
    for _ in range(100):
        res = policyConf.select_action(aggregate_bets)
        assert(res == expected)

@pytest.mark.parametrize("arr,vals,expected", [
    # random configurations with 6 actions start
    ([0,1,2,3,4,5],[[0],[0],[0],[0],[0],[100]],5),
    ([5,4,3,2,1,0],[[0],[0],[0],[0],[0],[100]],0),
    ([0,1,2,3,4,5],[[0],[10],[10000],[1000],[10000],[100]],2),
    ([5,4,3,2,1,0],[[100000],[10000],[1000],[100],[10],[0]],5),
    # random configurations with 6 actions end
    # more distributions for some actions start
    ([0,1,2],[[0],[0,1,2,3],[0]],None), # TODO: should fail
    # more distributions for some actions end
    # high reward later in the future start
    ([1,2],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,100000]],2),
    ([2,1],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,100000]],1),
    # high reward later in the future end
    # slight difference between actions start
    ([1,2],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,0.999]],1),
    ([2,1],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,0.999]],2),
    # slight difference between actions end
])
def test_suggested_policy_config_action_probs(arr,vals,expected,gen_policy_conf):
    """
    This test checks that action probs in policy config
    satsifies our definition of suggested general action probabilities which is identical
    to select action but gives out probabilities.

    [arr] is the list of actions available
    [vals] is the list of values for each action in [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    aggregate_bets : Dict[float,List[DiscreteDistribution]] = {}
    for k in range(len(arr)):
        aggregate_bets[arr[k]]= [constantDistribution(i) for i in vals[k]]
    res = policyConf.action_probabilities(aggregate_bets)
    for i in arr:
        if (i == expected):
            assert(floatIsEqual(res[i],1))
            continue
        assert(floatIsEqual(res[i],0))


class OneGoodOneBadDistribution(DiscreteDistribution):
    def __init__(self,constant:float):
        self.constant: float = constant
        self.good = False
    def sample(self) -> float:
        self.good = not(self.good)
        if (self.good):
            return self.constant
        return -1*self.constant

@pytest.mark.parametrize("arr,vals,expected", [
    # more distributions for some actions start
    ([0,1,2],[[0],[0,1,2,3],[0]],None), # TODO: should fail
    # more distributions for some actions end
    # high reward later in the future start
    ([1,2],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,100000]],1./2),
    ([2,1],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,100000]],1./2),
    # high reward later in the future end
    # slight difference between actions start
    ([1,2],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,0.999]],1./2),
    ([2,1],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,0.999]],1./2),
    # slight difference between actions end
    # identical actions start
    ([1,2],[[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]],1./2),
    # identical actions end
])
def test_suggested_policy_config_action_probs_uniform(arr,vals,expected,gen_policy_conf):
    """
    This test checks that action probs in policy config
    satsifies our definition of suggested general action probabilities which is identical
    to select action but gives out probabilities.

    [arr] is the list of actions available
    [vals] is the list of values for each action in [arr]
    The other parameter is a test fixture to help create
    objects easier.
    """
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    aggregate_bets : Dict[float,List[DiscreteDistribution]] = {}
    for k in range(len(arr)):
        aggregate_bets[arr[k]]= [OneGoodOneBadDistribution(i) for i in vals[k]]
    res = policyConf.action_probabilities(aggregate_bets)
    for i in arr:
        assert(floatIsEqual(res[i],expected,0.001))