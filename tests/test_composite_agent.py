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

@last-modified: 2021-01-02T20:30:47.974Z-06:00
"""

# standard library
# 3rd party packages
import pytest
# local source

@pytest.mark.parametrize("keys,lookup,expected", [
    ([()],{},0)
])
def test_configured_vote(keys,lookup,expected):
    pass

@pytest.mark.parametrize("keys,lookup,expected", [
    ([()],{},[0.0])
])
def test_configured_bet(keys,lookup,expected):
    pass

@pytest.mark.parametrize("keys,lookup,expected", [
    ([()],{},[0.0])
])
def test_configured_prediction(keys,lookup,expected):
    pass

@pytest.mark.parametrize("keys,lookup,expected", [
    ([()],{},0.0)
])
def test_should_fail_vote(keys,lookup,expected):
    pass

@pytest.mark.parametrize("keys,lookup,expected", [
    ([()],{},[0.0])
])
def test_should_fail_bet(keys,lookup,expected):
    pass

@pytest.mark.parametrize("keys,lookup,expected", [
    ([()],{},[0.0])
])
def test_should_fail_prediction(keys,lookup,expected):
    pass