# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-06 18:09:26
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:53:56


from tests.conftest import (
    pytest, project_types,
    factory as fac,
    vote_range,np,
    List, floatIsEqual
)

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


# @pytest.mark.parametrize("aggFun,VR,vals", [
#     (aggregateSimple, vote_range.BinaryVoteRange, [-1,0,0.5,1,2]),
#     (aggregateSimple, vote_range.BinaryVoteRange, [0 for _ in range(30)]),
#     (aggregateSimple, vote_range.BinaryVoteRange, [1 for _ in range(30)]),
#     (aggregateSimple, vote_range.BinaryVoteRange, []),
#     (aggregateSimple, vote_range.FiveStarVoteRange, [-1,0,1,2.5,3,5,6]),
#     (aggregateSimple, vote_range.FiveStarVoteRange, [5 for _ in range(30)]),
#     (aggregateSimple, vote_range.FiveStarVoteRange, [1 for _ in range(30)]),
#     (aggregateSimple, vote_range.FiveStarVoteRange, []),
#     (aggregateSimple, vote_range.ZeroToTenVoteRange, [-1,0.00000008,0,5,6.8,10,11]),
#     (aggregateSimple, vote_range.ZeroToTenVoteRange, [10 for _ in range(30)]),
#     (aggregateSimple, vote_range.ZeroToTenVoteRange, [0 for _ in range(30)]),
#     (aggregateSimple, vote_range.ZeroToTenVoteRange, []),
#     (aggregateSuggested, vote_range.BinaryVoteRange, [-1,0,0.5,1,2]),
#     (aggregateSuggested, vote_range.BinaryVoteRange, [0 for _ in range(30)]),
#     (aggregateSuggested, vote_range.BinaryVoteRange, [1 for _ in range(30)]),
#     (aggregateSuggested, vote_range.BinaryVoteRange, []),
#     (aggregateSuggested, vote_range.FiveStarVoteRange, [-1,0,1,2.5,3,5,6]),
#     (aggregateSuggested, vote_range.FiveStarVoteRange, [5 for _ in range(30)]),
#     (aggregateSuggested, vote_range.FiveStarVoteRange, [1 for _ in range(30)]),
#     (aggregateSuggested, vote_range.FiveStarVoteRange, []),
#     (aggregateSuggested, vote_range.ZeroToTenVoteRange, [-1,0.00000008,0,5,6.8,10,11]),
#     (aggregateSuggested, vote_range.ZeroToTenVoteRange, [10 for _ in range(30)]),
#     (aggregateSuggested, vote_range.ZeroToTenVoteRange, [0 for _ in range(30)]),
#     (aggregateSuggested, vote_range.ZeroToTenVoteRange, [])
# ])
# def test_policy_config(aggFun, VR,vals):
