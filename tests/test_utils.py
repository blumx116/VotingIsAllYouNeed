# -*- coding: utf-8 -*-
"""
Created on 2020-12-10 15:47:28
@author: suhail

This file tests general utils we are using
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

@pytest.mark.parametrize("dictionary",[
    ({'a':1,'b':2,'c':5}), # simple case
    ({str(a):a+1 for a in range(100)}), # general case
])
def test_dict_to_fn(dictionary):
    fun = U.dict_to_fn(dictionary)
    for i in dictionary:
        assert dictionary[i] == fun(i)

@pytest.mark.parametrize("dictionary,values",[
    ({'a':1,'b':2,'c':5},['d']), # key not present in dictionary
    ({'1':1,'2':2,'3':5},[1]), # key different type than one in dictionary
    ({},[]), # empty dictionary
    ({'a':1,'b':2,10:5},[]), # different types
    ({str(a):a+1 for a in range(100)}, ['100']), # general case
    ({str(a):a+1 for a in range(100)}, [99]), # general case
])
def test_dict_to_fn_should_fail(dictionary,values):
    try:
        fun = U.dict_to_fn(dictionary)
        for i in values:
            fun(i)
        assert(False)
    except:
        assert(True)