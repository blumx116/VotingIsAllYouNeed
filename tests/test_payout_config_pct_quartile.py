# -*- coding: utf-8 -*-
"""
file: test_payout_config_pct_quartile.py

@author: Suhail.Alnahari

@description: this file tests the 95% quartile payout mechanism

@created: 2020-12-20T15:12:52.489Z-06:00

@last-modified: 2020-12-20T16:40:26.841Z-06:00
"""

# standard library
from typing import Dict, List

# 3rd party packages
import pytest
import numpy as np

# local source
from tests.conftest import floatIsEqual
import VIAYN.project_types as P
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
from VIAYN.project_types import PayoutConfiguration, HistoryItem, A, S, ActionBet, Agent, WeightedBet
from VIAYN.samples.factory import PayoutConfigEnum as PCE, UpperBoundConfigEnum as UBCE


@pytest.mark.parametrize("enum,pred,t1,t0,R,expected", [
    (PCE.simple,[3],1,0,1,4), # single timestep predictions
    (PCE.simple,[0,3],2,0,1,4), # two timestep predictions
    (PCE.simple,[10,10,10,3,10,10,10],4,0,1,4), # multiple timestep good prediction
    (PCE.simple,[10,10,10,3,10,10,10],3,0,1,81), # multiple timestep bad prediction
    (PCE.suggested,[3],1,0,1,4),  # single timestep predictions
    (PCE.suggested,[0,3],2,0,1,4), # two timestep predictions
    (PCE.suggested,[10,10,10,3,10,10,10],4,0,1,4), # multiple timestep good prediction
    (PCE.suggested,[10,10,10,3,10,10,10],3,0,1,81), # multiple timestep bad prediction
])
def test_payout_config_calculate_loss(
    enum,pred,t1,t0,R,expected,
    gen_payout_conf, gen_weighted_bet
):
    """
    This test checks that calculate loss in payout config
    satisfies our definition of simple and suggested payouts
    for one agent is not affected by setting the new upperbound

    [enum] is the specifier for which payout config to use, given to the factory
    [wb] is the weighted bet the agent created at [t0] generated from [pred].
    The loss should be calculated for the [t1] weightedbet
    [R] is the welfare score.
    The other two parameters are test fixtures to help create
    objects easier.
    """
    pf: P.PayoutConfiguration = gen_payout_conf(
        enum, UBCE.quartile95
    )
    wb: P.WeightedBet  = gen_weighted_bet(pred,pred)
    assert(floatIsEqual(pf.calculate_loss(wb,t0,t1,R),expected))

@pytest.mark.parametrize("enum,bet,t1,t0,loss,allLs,aj,ai,expected", [
    # testing random bets and weights with main bet = 1
    (
        PCE.simple,1,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        1*1
    ),
    # testing random bets and weights with main bet = 5
    (
        PCE.simple,5,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        1*5
    ),
    # testing random bets and weights with high loss
    (
        PCE.simple,5,1,0,10,[(5,0),(4,1),(0.01,10)],1,1,
        0
    ),
    # testing random bets and weights with average loss
    (
        PCE.simple,5,1,0,5,[(5,0),(4,5),(0.01,10)],1,1,
        0
    ),
    # testing random weights and bets with equal losses
    (
        PCE.simple,5,1,0,5,[(5,5),(4,5),(0.01,5)],1,1,
        5
    ),
    # Checking behavior start ONLY 1 FROM EACH PAIR SHOULD PASS
    # 1st pair
    (
        PCE.simple,5,1,0,0,[(9,0),(5,2),(5,10)],1,1,
        5*2
    ),
    (
        PCE.simple,5,1,0,0,[(9,0),(5,2),(5,10)],1,1,
        5*10
    ),
    # 2nd pair
    (
        PCE.simple,5,1,0,0,[(9,0),(5.001,2),(4.999,10)],1,1,
        5*2
    ),
    (
        PCE.simple,5,1,0,0,[(9,0),(4.999,2),(5.001,10)],1,1,
        5*10
    ),
    # Checking behavior end
    # testing payout with one agent
    (
        PCE.simple,5,1,0,5,[(5,5)],1,1,
        5
    ),
    # testing random bets and weights with main bet = 1
    (
        PCE.suggested,1,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        1*(1/(1+(1-1)))
    ),
    # testing random bets and weights with main bet = 5
    (
        PCE.suggested,5,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        5*(1/(1+(1-1)))
    ),
    # testing random bets and weights with high loss
    (
        PCE.suggested,5,1,0,10,[(5,0),(4,1),(0.01,10)],1,1,
        0
    ),
    # testing random bets and weights with average loss
    (
        PCE.suggested,5,1,0,5,[(5,0),(4,5),(0.01,10)],1,1,
        5*(25/(25+(25-25)))
    ),
    # testing random weights and bets with equal losses
    (
        PCE.suggested,5,1,0,5,[(5,5),(4,5),(0.01,5)],1,1,
        5
    ),
    # testing payout with one agent
    (
        PCE.suggested,5,1,0,5,[(5,5)],1,1,
        5
    ),
    # Checking behavior start ONLY 1 FROM EACH PAIR SHOULD PASS
    # 1st pair
    (
        PCE.suggested,5,1,0,0,[(9,0),(5,2),(5,10)],1,1,
        5*2
    ),
    (
        PCE.suggested,5,1,0,0,[(9,0),(5,2),(5,10)],1,1,
        5*10
    ),
    # 2nd pair
    (
        PCE.suggested,5,1,0,0,[(9,0),(5.001,2),(4.999,10)],1,1,
        5*2
    ),
    (
        PCE.suggested,5,1,0,0,[(9,0),(4.999,2),(5.001,10)],1,1,
        5*10
    ),
    # Checking behavior end
])
def test_payout_config_calculate_payout_from_loss(
    enum,bet,t1,t0,loss,allLs,aj,ai,expected,
    gen_payout_conf, gen_weighted_bet
):
    """
    This test checks that calculate payout for a given loss in payout 
    config satisfies our definition of simple and suggested payouts
    for one agent is not affected by setting the new upperbound

    [enum] is the specifier for which payout config to use, given to the factory
    [bet] is the money the agent bet at [t1].
    The [loss] is specified at [t1] for a certain agent.
    [allLs] are the weights and bet amounts of all agents at [t1]
    [aj] is the action that occured and [ai] is the action in question
    The other two parameters are test fixtures to help create
    objects easier.
    """
    pf: P.PayoutConfiguration = gen_payout_conf(
        enum, UBCE.quartile95
    )
    assert(
        floatIsEqual(
            pf.calculate_payout_from_loss(
                bet,loss,allLs,t0,t1,aj,ai
            ),
            expected
        )
    )

@pytest.mark.parametrize("enum,welfare_score,selectedA, t_current,weightedBets,expected", [
    # These test cases are mostly sanity checks with random configurations of
    # a three agent system. The configurations were set to have a sample of what could happen
    # but are not exhaustive. 
    (
        PCE.simple,
        5.5,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 50.0,'A2':0.}
    ),
    (
        PCE.simple,
        5.5,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':8.4}
    ),
    (
        PCE.simple,
        7,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':6.0}
    ),
    (
        PCE.simple,
        9,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.0 ,'A2':0.75}
    ),
    (
        PCE.simple,
        8,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':1.05}
    ),
    (
        PCE.simple,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 2,'A2':1.2}
    ),
    (
        PCE.simple,
        5.5,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 50.0,'A2':0., 'A3': 84.0}
    ),
    (
        PCE.simple,
        5.5,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 483,'A2': 201.6, 'A3': 0.0}
    ),
    (
        PCE.simple,
        7,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.,'A2':6.0,'A3':30.375}
    ),
    (
        PCE.simple,
        9,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 4.0625,'A2':1.2375, 'A3': 0.0}
    ),
    (
        PCE.simple,
        8,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 60,'A2':8.25, 'A3': 0.0}
    ),
    (
        PCE.simple,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 5.12,'A2': 3.072,'A3': 0.0}
    ),
    (
        PCE.suggested,
        3,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 2.5,'A2':2.7}
    ),
    (
        PCE.suggested,
        4,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':2.1}
    ),
    (
        PCE.suggested,
        4.5,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':2.7}
    ),
    (
        PCE.suggested,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'), # division by zero
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'), # division by zero
        ],
        {'A1': 2.0,'A2':1.2}
    ),
    (
        PCE.suggested,
        7.5,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 1.4,'A2':0.}
    ),
    (
        PCE.suggested,
        0,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 1.4,'A2':0.}
    ),
    (
        PCE.suggested,
        1,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'), # divide by zero
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'), # divide by zero
        ],
        {'A1': 2.0,'A2':1.2}
    ),
    (
        PCE.suggested,
        5.5,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 4.179104478 ,'A2':0., 'A3': 7.020895522}
    ),
    (
        PCE.suggested,
        5.5,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 1.587423313,'A2': 0.662576687, 'A3': 0.0}
    ),
    (
        PCE.suggested,
        7,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.,'A2':1.187628866,'A3':6.012371134}
    ),
    (
        PCE.suggested,
        9,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 2.222877358,'A2':0.677122642, 'A3': 0.0}
    ),
    (
        PCE.suggested,
        8,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 2.549450549,'A2':0.350549451, 'A3': 0.0}
    ),
    (
        PCE.suggested,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 5.75,'A2': 3.45,'A3': 0.0}
    ),
])
def test_payout_config_calculate_all_payouts(
    enum, # factory spec variable
    welfare_score,
    selectedA, t_current, # history
    weightedBets,
    expected,
    gen_payout_conf, gen_weighted_bet,gen_history_item # fixtures
):
    """
    This test checks that calculate all payouts for a given history item
    in payout config satisfies our definition of simple and suggested payouts
    for one agent is not affected by setting the new upperbound

    [enum] is the specifier for which payout config to use, given to the factory
    [welfare_score] is the total happiness for all agents at [t_current].
    [selectedA] is the action selected by all agents at [t_current]
    [weightedBets] is what the agents bet at [t_current]
    The other three parameters are test fixtures to help create
    objects easier.
    """
    
    pf: P.PayoutConfiguration = gen_payout_conf(
        enum, UBCE.quartile95
    )
    predsDict: Dict[A, List[WeightedBet[A, S]]] = {}
    
    # creating list of weighted bets at t1
    for bet,pred,action,money,castby in weightedBets:
        if (action not in predsDict.keys()):
            predsDict[action] = []
        predsDict[action].append(
            gen_weighted_bet(
                bet,pred,action,money,castby
            )
        )
    # TODO: doesn't hit t_cast_on != 0
    # creating history item that's created at t0 = 0
    record: HistoryItem[A,S] = gen_history_item(
        selectedA,
        predsDict,
        0
    )

    # calculting payouts for history item
    payouts = pf.calculate_all_payouts(
        record,
        welfare_score,
        t_current,
    )
    for agent in payouts:
        assert(
            floatIsEqual(
                payouts[agent],
                expected[agent]
            )
        )
