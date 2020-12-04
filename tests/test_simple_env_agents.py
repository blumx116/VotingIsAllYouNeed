# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 19:25:18
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-03 20:39:10


from conftest import pytest, project_types


@pytest.mark.parametrize("AB, vote", [
    (project_types.ActionBet([0.],[5.]) , 5.),
    (project_types.ActionBet([0.],[4.]) , 5.),
    (project_types.ActionBet([0.5],[1.]) , 5.),
    (project_types.ActionBet([0.5],[0.]), 0),
    (project_types.ActionBet([0.5],[2.]), 2),
    (project_types.ActionBet([0.5],[2.]), 1),
    (project_types.ActionBet([0.5],[7.]), 10),
    (project_types.ActionBet([0.],[4.]), 9),
    (project_types.ActionBet([0.],[2.]), 2),
    (project_types.ActionBet([1],[5.]), 5),
    (project_types.ActionBet([1.],[0.]), 0),    
])
def test_constant_agent_full(AB,vote):
    print(f"consant: {AB.bet} and {AB.prediction}, {vote}")
    assert(True)

@pytest.mark.parametrize("vote, seed", [
    (5.,0),
    (5., 123),    
    (0., 1231131),
    (1., 1235234),
    (2., 5124),
    (8., 82384),
    (10., 0),
])
def test_random_agent_full(vote, seed):
    print(f"random: {vote} , {seed}")
    assert(True)
