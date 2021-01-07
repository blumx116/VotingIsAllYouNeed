# -*- coding: utf-8 -*-
"""
file: test_composite_agent.py

@author: Suhail.Alnahari

@description: file that tests that a composite agent with a defined 
              dictionary defined for voting, betting, or predicting mechanism
              behaves as expected.
              To configure a bet, vote, or prediction mechanism, we use the
              dictionary:
                    {
                        (None,None,None): <default value>
                    } 
              to always choose the <default value> whenever a bet, vote, or 
              prediction is required.

@created: 2021-01-02T18:46:30.736Z-06:00

@last-modified: 2021-01-07T12:58:51.861Z-06:00
"""

# standard library
from typing import (
    List, Dict
)
# 3rd party packages
import pytest

# local source
from VIAYN.project_types import (
    Agent, S
)
from VIAYN.samples.factory import (
    AgentsEnum
)
from tests.conftest import floatIsEqual


@pytest.mark.parametrize("keys,lookup", [
    # unique states with unique votes
    ([i for i in range(100)],{i:i+1 for i in range(100)})
])
def test_configured_vote(keys,lookup,gen_agent):
    """
    Test that checks if look-up voting works for a composite agent

    keys: List[S]
        states to vote for
    lookup: Dict[S,int]
        vote look-up for each state
    gen_agent: fn
        factory fixture
    """
    a: Agent = gen_agent(
        AgentsEnum.composite, 20,
        vote_lookup=lookup
    )
    for key in keys:
        assert a.vote(key) == (key+1)

@pytest.mark.parametrize("N,key,expected", [
    (100,('a',1,4),0.3),
    (4,('a',2,5),0.4),
    (10,('a',2,10),0.6),
    (1,('a',2,4),0.1),
    (50,('d',3,1),0.0),
])
def test_configured_bet(N,key,expected,gen_agent):
    """
    Test that checks if look-up ActionBets work for a composite agent

    The tests inputs floats and expect a list of bets/predictions
    with length N. keyBet and keyPred have identical keys but values of
    keyPred are 10*values of keyBet to remove percent property. Expected
    is based on keyBet so we multiply by 10 to compare keyPred.

    N: int
        length of bet/prediction
    key: Tuple(S,A,money)
        what happens when an agent sees key
    expected: float
        value of bet at key and value/10 of prediction at key
    gen_agent: fn
        factory fixture
    """
    keyBet: Dict = {
        (None,None,None):0.0,
        ('a',None,None):0.1,
        ('a',1,None):0.2,
        ('a',1,4):0.3,
        (None,2,5):0.4,
        (None,None,10):0.5,
        (None,2,10):0.6,
        ('a',1,1):0.7,
    }
    keyPred: Dict = {
        (None,None,None):0,
        ('a',None,None):1,
        ('a',1,None):2,
        ('a',1,4):3,
        (None,2,5):4,
        (None,None,10):5,
        (None,2,10):6,
        ('a',1,1):7,
    }
    a: Agent = gen_agent(
        AgentsEnum.composite, 20,
        bet_lookup=keyBet,
        prediction_lookup=keyPred,
        N=N
    )
    assert len(a.bet(key[0],key[1],key[2]).bet) == N 
    assert len(a.bet(key[0],key[1],key[2]).prediction) == N 
    assert floatIsEqual(sum(a.bet(key[0],key[1],key[2]).bet), expected)
    assert floatIsEqual(sum(a.bet(key[0],key[1],key[2]).prediction), expected*N*10)

# @pytest.mark.parametrize("keys,lookup,expected", [
#     ([()],{},[0.0])
# ])
# def test_configured_prediction(keys,lookup,expected,gen_agent):
#     pass

# @pytest.mark.parametrize("keys,lookup,expected", [
#     ([()],{},0.0)
# ])
# def test_should_fail_vote(keys,lookup,expected,gen_agent):
#     pass

# @pytest.mark.parametrize("keys,lookup,expected", [
#     ([()],{},[0.0]),
#     (('d',3,'f'),None),
# ])
# def test_should_fail_bet(keys,lookup,expected,gen_agent):
#     pass

# @pytest.mark.parametrize("keys,lookup,expected", [
#     ([()],{},[0.0])
# ])
# def test_should_fail_prediction(keys,lookup,expected,gen_agent):
#     pass