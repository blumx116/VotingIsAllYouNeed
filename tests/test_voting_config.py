# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-04 22:06:47
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-11 23:17:59

from typing import List
from tests.conftest import floatIsEqual
import pytest
import VIAYN.project_types as project_types
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import numpy as np

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


@pytest.mark.parametrize("aggFun,VR,vals", [
    (aggregateSimple, vote_range.BinaryVoteRange, [-1,0,0.5,1,2]),
    (aggregateSimple, vote_range.BinaryVoteRange, [0 for _ in range(30)]),
    (aggregateSimple, vote_range.BinaryVoteRange, [1 for _ in range(30)]),
    (aggregateSimple, vote_range.BinaryVoteRange, []),
    (aggregateSimple, vote_range.FiveStarVoteRange, [-1,0,1,2.5,3,5,6]),
    (aggregateSimple, vote_range.FiveStarVoteRange, [5 for _ in range(30)]),
    (aggregateSimple, vote_range.FiveStarVoteRange, [1 for _ in range(30)]),
    (aggregateSimple, vote_range.FiveStarVoteRange, []),
    (aggregateSimple, vote_range.ZeroToTenVoteRange, [-1,0.00000008,0,5,6.8,10,11]),
    (aggregateSimple, vote_range.ZeroToTenVoteRange, [10 for _ in range(30)]),
    (aggregateSimple, vote_range.ZeroToTenVoteRange, [0 for _ in range(30)]),
    (aggregateSimple, vote_range.ZeroToTenVoteRange, []),
    (aggregateSuggested, vote_range.BinaryVoteRange, [-1,0,0.5,1,2]),
    (aggregateSuggested, vote_range.BinaryVoteRange, [0 for _ in range(30)]),
    (aggregateSuggested, vote_range.BinaryVoteRange, [1 for _ in range(30)]),
    (aggregateSuggested, vote_range.BinaryVoteRange, []),
    (aggregateSuggested, vote_range.FiveStarVoteRange, [-1,0,1,2.5,3,5,6]),
    (aggregateSuggested, vote_range.FiveStarVoteRange, [5 for _ in range(30)]),
    (aggregateSuggested, vote_range.FiveStarVoteRange, [1 for _ in range(30)]),
    (aggregateSuggested, vote_range.FiveStarVoteRange, []),
    (aggregateSuggested, vote_range.ZeroToTenVoteRange, [-1,0.00000008,0,5,6.8,10,11]),
    (aggregateSuggested, vote_range.ZeroToTenVoteRange, [10 for _ in range(30)]),
    (aggregateSuggested, vote_range.ZeroToTenVoteRange, [0 for _ in range(30)]),
    (aggregateSuggested, vote_range.ZeroToTenVoteRange, [])
])
def test_vote_config(aggFun, VR,vals,gen_vote_conf):
    vc: project_types.VotingConfiguration = gen_vote_conf(
        fac.VotingConfigEnum.simple,
        VR
    )
    assert(vc.n_agents == 0)
    vc.set_n_agents(len(vals))
    assert(vc.n_agents == len(vals))
    assert(vc.max_possible_vote_total() == (len(vals)*VR.maxVote()))
    assert(vc.min_possible_vote_total() == (len(vals)*VR.minVote()))
    assert(floatIsEqual(vc.aggregate_votes(vals), aggFun(vals,VR)))    

