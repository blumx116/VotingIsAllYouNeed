# -*- coding: utf-8 -*-
"""
Created on 2020-12-10 15:47:28
@author: suhail

This file tests general utils we are using
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


@pytest.mark.parametrize("dict1,dict2,expected", [
    ({'a1':1,'a2':2},{},{'a1':1,'a2':2}), # identical dicts
    ({},{'a1':1,'a2':2},{'a1':1,'a2':2}), # one empty dict
    ({'a1':1,'a2':2},{},{'a1':1,'a2':2}), # the other is empty
    ({'a1':1,'a2':2},{'a3':1,'a4':2},{'a1':1,'a2':2,'a3':1,'a4':2}), # unique dicts
    ({},{},{}), # dicts are all empty
])
def test_add_dictionaries(dict1,dict2,expected):
    res : dict = U.add_dictionaries(dict1,dict2)
    assert(len(res.keys()) == len(expected.keys()))
    for i in res.keys():
        assert(floatIsEqual(res[i],expected[i]))


@pytest.mark.parametrize("bets,predictions,moneys,expected", [
    # random configurations of a three agent system with identical money start
    # using one time-step predictions 
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
    # random configurations of a three agent system with identical money end
    # using one time-step predictions

    # random configurations of a three agent system with identical money start
    # using two time-step predictions
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
    # random configurations of a three agent system with identical money end
    # using two time-step predictions
    
    # random configurations of a three agent system with varying money start
    # using one time-step predictions
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
    # random configurations of a three agent system with varying money end
    # using one time-step predictions

    # random configurations of a three agent system with varying money start
    # using two time-step predictions
    (
        [[0.2,0.2],[0.3,0.3],[0.2,0.2]],
        [[2,2],[4,4],[5,5]],
        [2,3,4],
        [(0.8/2.1)+((0.9*4)/2.1)+(4./2.1), (0.8/2.1)+((0.9*4)/2.1)+(4./2.1)]
    )
    # random configurations of a three agent system with varying money end
    # using two time-step predictions
])
def test_weighted_mean(bets,predictions,moneys,expected,gen_weighted_bet):
    """
    Checks that given a set of agent [bets] and [predictions], the weighted
    mean of them is equal to sum b_it p_it m_i / sum b_i m_i per time-step
    
    [bets] b_it
    [predictions] p_it
    [moneys] m_i
    The other parameter is a test fixture to help create
    objects easier.
    """
    weightedBets : List[P.WeightedBet] = []
    for i in range(len(bets)):
        print(bets[i],predictions[i],moneys[i])
        weightedBets.append(
            gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
        )
    res = U.weighted_mean_of_bets(weightedBets)
    assert(len(res) == len(expected))
    for i in range(len(res)):
        assert(floatIsEqual(res[i],expected[i]))


@pytest.mark.parametrize("bets,predictions,moneys,expected", [
    # zero/zero
    (
        [[0]],
        [[0]],
        [2],
        [0]
    ),
    # bets length is not equal to prediction length
    (
        [[0,2,3]],
        [[0]],
        [2],
        [0]
    ),
    # prediction is empty
    (
        [[0,2,3]],
        [[]],
        [2],
        [0]
    ),
    # no money
    (
        [[1,2,3]],
        [[3,2,3]],
        [],
        [0]
    ),
    # empty bets and predictions
    (
        [],
        [],
        [2],
        []
    ),
    # varying lengths per agent
    (
        [[1,2,3],[1,2],[1,2,3]],
        [[1,2,3],[1,2],[1,2,3]],
        [2,3,4],
        [0,2,3]
    ),
    # varying lengths per agent
    (
        [[1,2],[1,2],[1,2,3]],
        [[1,2],[1,2],[1,2,3]],
        [2,3,4],
        [0,2,3]
    )
])
def test_weighted_mean_shouldFail(bets,predictions,moneys,expected,gen_weighted_bet):
    """
    Checks that given a set of agent [bets] and [predictions], the weighted
    mean of them is equal to sum b_it p_it m_i / sum b_i m_i per time-step
    cannot be calculated given the parameters input
    
    [bets] b_it
    [predictions] p_it
    [moneys] m_i
    The other parameter is a test fixture to help create
    objects easier.
    """
    try:
        weightedBets : List[P.WeightedBet] = []
        for i in range(len(bets)):
            print(bets[i],predictions[i],moneys[i])
            weightedBets.append(
                gen_weighted_bet(bets[i],predictions[i],money=moneys[i])
            )
        
        res = U.weighted_mean_of_bets(weightedBets)
        assert(False)
    except:
        assert(True)


@pytest.mark.parametrize("arr,vals,expected", [
    ([0,1,2,3,4,5],[0,0,0,0,0,100],5), # last item is the maximum
    ([5,4,3,2,1,0],[0,0,0,0,0,100],5), # first item is the maximum
    ([0,1,2,3,4,5],[0,10,100,1000,10000,100000],5), # varying values
    ([5,4,3,2,1,0],[100000,10000,1000,100,10,0],0), # order doesn't matter
    ([],[],None), # empty values
    ([0,1,2,3,4],[1,1,1,1,1],0), # first item with identical value is returned #TODO: document this behavior
    ([1,1,1,1,1,1],[7,0,7,7,7,7,7,7],1), # items with the same value return the same thing
])
def test_argmax(arr,vals,expected):
    assert(U.argmax(arr,lambda x: vals[x]) == expected)


@pytest.mark.parametrize("item,filter,expected", [
    # All items in item are non-None
    (('a',1,'b'),(None,None,None),0), # nothing matches
    (('a',1,'b'),('a',None,None),1), # 1 matches
    (('a',1,'b'),('a',1,None),2), # 2 matches
    (('a',1,'b'),('a',1,'b'),3), # All match
    (('a',1,'b'),(None,1,'b'),2), # 2 matches backwards
    (('a',1,'b'),(None,None,'b'),1), # 1 match backwards
    (('a',1,'b'),(None,1,None),1), # 1 match

    # There exists a None
    ((None,None,None),(None,None,None),0), # nothing matches
    (('a',None,'b'),('a',None,None),1), # 1 matches
    (('a',1,None),('a',1,None),2), # 2 matches
    ((None,1,'b'),(None,1,'b'),2), # 2 matches backwards
    ((None,None,'b'),(None,None,'b'),1), # 1 match backwards
    ((None,1,None),(None,1,None),1), # 1 match
])
def test_iterable_matches(item,filter,expected):
    """
    Checks that [U.iterable_matches] works as documented.

    item: Sequence
        item to check
    filter: Sequence
        possible values that could be matched to [item]
    expected: int
        expected n_matches from the result of [U.iterable_matches]
    """
    assert U.iterable_matches(item,filter) == expected

@pytest.mark.parametrize("key,expected", [
    (('a', 1,'b'),3),
    (('a',2,'c'),4),
    (('a',2,'d'),6),
    (('a',2,'b'),1),
    (('d',3,'f'),0),
    (('d',3,'k'),0),
])
def test_behaviour_lookup_from_dict(key,expected):
    """
    Checks that [U.behaviour_lookup_from_dict] works as documented.

    key: Sequence
        item to check
    expected: int
        expected best_matching_val from the result of [U.behaviour_lookup_from_dict]
    """
    keyVal: Dict = {
        (None,None,None):0,
        ('a',None,None):1,
        ('a',1,None):2,
        ('a',1,'b'):3,
        (None,2,'c'):4,
        (None,None,'d'):5,
        (None,2,'d'):6,
        ('a',1,'k'):7
    }
    assert U.behaviour_lookup_from_dict(key,keyVal) == expected