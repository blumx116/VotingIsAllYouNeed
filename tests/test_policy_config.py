# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-06 18:09:26
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:59:45


from tests.conftest import (
    pytest, project_types,
    factory as fac,
    vote_range,np,
    List, floatIsEqual
)

def aggregateSimple(
    weights: List[List[float]],
    predictions: List[List[float]]
    ) -> List[float]:
    res: List[float] = []
    ws = np.asarray(weights)
    ps = np.asarray(predictions)
    print(ws.shape)
    print(ps.shape)
    assert(ws.shape() == ps.shape())
    for j in range(ws.shape()[1]):
        actionSum: float = 0.
        for i in range(ws.shape()[0]):
            actionSum += (ws[i,j]*ps[i,j])
        res.append(actionSum/sum(ws[:,j]))
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
