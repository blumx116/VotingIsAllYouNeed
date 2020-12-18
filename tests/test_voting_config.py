# -*- coding: utf-8 -*-
"""
Created on 2020-12-04 22:06:47
@author: suhail

This file tests voting config
"""
# standard library
from typing import List

# 3rd party packages
import pytest
import numpy as np

# local source
from tests.conftest import floatIsEqual
import VIAYN.project_types as project_types
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range

def aggregateSimple(
    votes: List[float],
    vr:project_types.VoteRange
    ) -> float:
    res: float = 0.
    for i in votes:
        if (vr.contains(i)):
            res += i
    return res

def aggregateSuggested(
    votes: List[float],
    vr:project_types.VoteRange
    ) -> float:
    res: float = 0.
    for i in votes:
        if (vr.contains(i)):
            res += i**0.5
    return res


@pytest.mark.parametrize("aggFun,VR,vals,spec_enum", [
    # random simple case of votes for a specific config
    (aggregateSimple, vote_range.BinaryVoteRange(), [-1,0,0.5,1,2], fac.VotingConfigEnum.simple),
    # extreme case
    (aggregateSimple, vote_range.BinaryVoteRange(), [0 for _ in range(30)], fac.VotingConfigEnum.simple),
    # extreme case
    (aggregateSimple, vote_range.BinaryVoteRange(), [1 for _ in range(30)], fac.VotingConfigEnum.simple),
    # empty
    (aggregateSimple, vote_range.BinaryVoteRange(), [], fac.VotingConfigEnum.simple),
    
    # random simple case of votes for a specific config
    (aggregateSimple, vote_range.FiveStarVoteRange(), [-1,0,1,2.5,3,5,6], fac.VotingConfigEnum.simple),
    # extreme case
    (aggregateSimple, vote_range.FiveStarVoteRange(), [5 for _ in range(30)], fac.VotingConfigEnum.simple),
    # extreme case
    (aggregateSimple, vote_range.FiveStarVoteRange(), [1 for _ in range(30)], fac.VotingConfigEnum.simple),
    # empty
    (aggregateSimple, vote_range.FiveStarVoteRange(), [], fac.VotingConfigEnum.simple),
    
    # random simple case of votes for a specific config
    (aggregateSimple, vote_range.ZeroToTenVoteRange(), [-1,0.00000008,0,5,6.8,10,11], fac.VotingConfigEnum.simple),
    # extreme case
    (aggregateSimple, vote_range.ZeroToTenVoteRange(), [10 for _ in range(30)], fac.VotingConfigEnum.simple),
    # extreme case
    (aggregateSimple, vote_range.ZeroToTenVoteRange(), [0 for _ in range(30)], fac.VotingConfigEnum.simple),
    # empty
    (aggregateSimple, vote_range.ZeroToTenVoteRange(), [], fac.VotingConfigEnum.simple),
    
    # random simple case of votes for a specific config
    (aggregateSuggested, vote_range.BinaryVoteRange(), [-1,0,0.5,1,2], fac.VotingConfigEnum.suggested),
    # extreme case
    (aggregateSuggested, vote_range.BinaryVoteRange(), [0 for _ in range(30)], fac.VotingConfigEnum.suggested),
    # extreme case
    (aggregateSuggested, vote_range.BinaryVoteRange(), [1 for _ in range(30)], fac.VotingConfigEnum.suggested),
    # empty
    (aggregateSuggested, vote_range.BinaryVoteRange(), [], fac.VotingConfigEnum.suggested),
    
    # random simple case of votes for a specific config
    (aggregateSuggested, vote_range.FiveStarVoteRange(), [-1,0,1,2.5,3,5,6], fac.VotingConfigEnum.suggested),
    # extreme case
    (aggregateSuggested, vote_range.FiveStarVoteRange(), [5 for _ in range(30)], fac.VotingConfigEnum.suggested),
    # extreme case
    (aggregateSuggested, vote_range.FiveStarVoteRange(), [1 for _ in range(30)], fac.VotingConfigEnum.suggested),
    # empty
    (aggregateSuggested, vote_range.FiveStarVoteRange(), [], fac.VotingConfigEnum.suggested),
    
    # random simple case of votes for a specific config
    (aggregateSuggested, vote_range.ZeroToTenVoteRange(), [-1,0.00000008,0,5,6.8,10,11], fac.VotingConfigEnum.suggested),
    # extreme case
    (aggregateSuggested, vote_range.ZeroToTenVoteRange(), [10 for _ in range(30)], fac.VotingConfigEnum.suggested),
    # extreme case
    (aggregateSuggested, vote_range.ZeroToTenVoteRange(), [0 for _ in range(30)], fac.VotingConfigEnum.suggested),
    (aggregateSuggested, vote_range.ZeroToTenVoteRange(), [], fac.VotingConfigEnum.suggested)
])
def test_vote_config(aggFun, VR,vals,gen_vote_conf, spec_enum):
    """
    Checks that votes are aggregated correctly in voting config that
    are validated using [aggFun] with the specific [VR] specified

    [aggFun] function that aggregates votes that is defined in this file.
    [VR] voting range specified
    [vals] votes to be aggregated
    [spec_enum] voting configuration specifier
    The other parameter is a test fixture to help create
    objects easier.
    """
    vc: project_types.VotingConfiguration = gen_vote_conf(
        spec_enum,
        VR
    )
    assert(vc.n_agents == 0)
    vc.set_n_agents(len(vals))
    assert(vc.n_agents == len(vals))
    if np.isfinite(vc.max_possible_vote_total()):
        assert floatIsEqual(vc.max_possible_vote_total(), aggFun([VR.maxVote()] * len(vals), VR))
    else:
        assert not np.isfinite(VR.maxVote())
    if np.isfinite(vc.min_possible_vote_total()):
        assert floatIsEqual(vc.min_possible_vote_total(), aggFun([VR.minVote()] * len(vals), VR))
    else:
        assert not np.isfinite(VR.minVote())
    assert vc.min_possible_vote_total() <= vc.max_possible_vote_total()
    assert(floatIsEqual(vc.aggregate_votes(vals), aggFun(vals,VR)))    

