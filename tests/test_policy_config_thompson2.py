# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-06 18:09:26
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 23:49:15

from typing import List
from tests.conftest import floatIsEqual
import pytest
import VIAYN.project_types as project_types
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import numpy as np
import VIAYN.utils as U



@pytest.mark.parametrize("bets,predictions,moneys,expected", [
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
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1), (0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    )
])
def test_suggested_policy_config_single_action(
    bets,predictions,
    moneys,expected,
    gen_policy_conf,gen_weighted_bet):
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    weightedBets : List[project_types.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {'a':weightedBets}
    res = policyConf.aggregate_bets(dic)['a']
    

@pytest.mark.parametrize("bets,predictions,moneys,expected", [
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
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1), (0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    )
])
def test_suggested_policy_config_multiple_actions(
    bets,predictions,
    moneys,expected,
    gen_policy_conf,gen_weighted_bet):
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    weightedBets : List[project_types.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    dic = {i:weightedBets for i in range(100)}
    res = policyConf.aggregate_bets(dic)
    


@pytest.mark.parametrize("arr,vals,expected", [
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5),
    ([5,4,3,2,1,0],[0,0,0,0,0,100],5),
    ([0,1,2,3,4,5],[0,10,100,1000,10000,100000],5),
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],0),
    ([],[],None),
])
def test_suggested_policy_config_select_action(arr,vals,expected,gen_policy_conf):
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    res = policyConf.select_action({key:val for key,val in zip(arr,vals)})

@pytest.mark.parametrize("arr,vals,expected", [
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5),
    ([5,4,3,2,1,0],[0,0,0,0,0,100],5),
    ([0,1,2,3,4,5],[0,10,100,1000,10000,100000],5),
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],0),
    ([],[],None),
])
def test_suggested_policy_config_action_probs(arr,vals,expected,gen_policy_conf):
    policyConf = gen_policy_conf(fac.PolicyConfigEnum.suggested_general)
    res = policyConf.action_probabilities({key:val for key,val in zip(arr,vals)})
    
