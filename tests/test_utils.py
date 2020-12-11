# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-10 15:47:28
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 23:36:41
from typing import List
from tests.conftest import floatIsEqual
import pytest
import VIAYN.project_types as P
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import numpy as np
import VIAYN.utils as U


@pytest.mark.parametrize("dict1,dict2,expected", [
    ({'a1':1,'a2':2},{},{'a1':1,'a2':2}),
    ({},{'a1':1,'a2':2},{'a1':1,'a2':2}),
    ({'a1':1,'a2':2},{'a3':1,'a4':2},{'a1':1,'a2':2,'a3':1,'a4':2}),
    ({},{},{}),
])
def test_add_dictionaries(dict1,dict2,expected):
    res : dict = U.add_dictionaries(dict1,dict2)
    assert(len(res.keys()) == len(expected.keys()))
    for i in res.keys():
        assert(floatIsEqual(res[i],expected[i]))


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
def test_weighted_mean(bets,predictions,moneys,expected,gen_weighted_bet):
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    res = U.weighted_mean(weightedBets)
    assert(len(res) == len(expected))
    for i in range(len(res)):
        assert(floatIsEqual(res[i],expected[i]))


@pytest.mark.parametrize("bets,predictions,moneys,expected", [
    (
        [[0]],
        [[0]],
        [2],
        [0]
    ),
    (
        [[0,2,3]],
        [[0]],
        [2],
        [0]
    ),
    (
        [[0,2,3]],
        [[]],
        [2],
        [0]
    ),
    (
        [[1,2,3]],
        [[3,2,3]],
        [],
        [0]
    ),
    (
        [],
        [],
        [2],
        []
    ),
    (
        [[1,2,3],[1,2],[1,2,3]],
        [[1,2,3],[1,2],[1,2,3]],
        [2,3,4],
        [0,2,3]
    ),
    (
        [[1,2],[1,2],[1,2,3]],
        [[1,2],[1,2],[1,2,3]],
        [2,3,4],
        [0,2,3]
    )
])
def test_weighted_mean_shouldFail(bets,predictions,moneys,expected,gen_weighted_bet):
    try:
        weightedBets : List[P.WeightedBet] = []
        for i in range(len(bets)):
            print(bets[i],predictions[i],moneys[i])
            weightedBets.append(
                gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
            )
        
        res = U.weighted_mean(weightedBets)
        assert(False)
    except:
        assert(True)


@pytest.mark.parametrize("arr,vals,expected", [
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5),
    ([5,4,3,2,1,0],[0,0,0,0,0,100],5),
    ([0,1,2,3,4,5],[0,10,100,1000,10000,100000],5),
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],0),
    ([],[],None),
    ([0,1,2,3,4],[1,1,1,1,1],0), # expected behavior?
    ([1,1,1,1,1,1],[7,0,7,7,7,7,7,7],1), # expected behavior?
    ([0],[],None) # expected behavior?
])
def test_argmax(arr,vals,expected):
    assert(U.argmax(arr,lambda x: vals[x]) == expected)