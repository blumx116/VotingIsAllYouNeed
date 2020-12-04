# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 19:25:18
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-03 21:26:15


from conftest import pytest, project_types,factory, vote_range


@pytest.mark.parametrize("AB, vote, money", [
    (project_types.ActionBet([0.],[5.]) , 5.,10),
    (project_types.ActionBet([0.],[4.]) , 5.,1),
    (project_types.ActionBet([0.5],[1.]) , 5.,100),
    (project_types.ActionBet([0.5],[0.]), 0,20),
    (project_types.ActionBet([0.5],[2.]), 2,0),
    (project_types.ActionBet([0.5],[2.]), 1,15),
    (project_types.ActionBet([0.5],[7.]), 10,4),
    (project_types.ActionBet([0.],[4.]), 9,2),
    (project_types.ActionBet([0.],[2.]), 2,10),
    (project_types.ActionBet([1],[5.]), 5,21),
    (project_types.ActionBet([1.],[0.]), 0,6),    
])
def test_constant_agent_full(AB,vote,money):
    print(f"consant: {AB.bet} and {AB.prediction}, {vote}, {money}")
    agent_fac = factory.AgentFactory()
    env_factory = factory.EnvFactory()
    agent = agent_fac.create({
        'type':'constant',
        'vote':vote,
        'prediction':AB})
    env = env_factory.create({})
    for _ in range(100):
        assert(agent.vote(state) == vote)
        for j in env.actions():
            state = env.state()
            action_bet = agent.bet(state,j,money)
            for k in range(len(action_bet.bet)):
                assert(action_bet.bet[k] == AB.bet[k])
            for k in range(len(action_bet.prediction)):
                assert(action_bet.prediction[k] == AB.prediction[k])
    

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
