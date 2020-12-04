# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 19:25:18
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-04 11:35:09


from conftest import pytest, project_types,factory, vote_range,np

def floatIsEqual(num1: float,num2: float) -> bool:
    return abs(num1-num2) < 0.000001

def test_constant_simple_agent_basic(constant_agent_config):
    for AB,vote in constant_agent_config:
        print(f"constant: {AB.bet} and {AB.prediction}, {vote}")
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
                action_bet = agent.bet(state,j,1)
                assert(
                    len(action_bet.bet) == len(AB.bet)
                )
                assert(
                    len(action_bet.prediction) == len(AB.prediction)
                )
                for k in range(len(action_bet.bet)):
                    assert(floatIsEqual(action_bet.bet[k], AB.bet[k]))
                for k in range(len(action_bet.prediction)):
                    assert(floatIsEqual(action_bet.prediction[k], AB.prediction[k]))
    

def test_random_simple_agent_basic(random_agent_config):
    for vote,seed in random_agent_config:
        print(f"random: {vote} , {seed}")
        agent_fac = factory.AgentFactory()
        env_factory = factory.EnvFactory()
        votingConfigFac = factory.VotingConfigFactory()
        votingConf = votingConfigFac.create({})
        agent = agent_fac.create({
            'type':'random',
            'vote':vote,
            'N': 1,
            'seed': seed,
            'totalVotesBound': (
                votingConf.min_possible_vote_total,
                votingConf.max_possible_vote_total)
            })
        env = env_factory.create({})
        genBet = np.random.default_rng(seed=seed)
        genPred = np.random.default_rng(seed=seed)
        for _ in range(100):
            assert(agent.vote(state) == vote)
            for j in env.actions():
                state = env.state()
                action_bet = agent.bet(state,j,1)
                for k in range(len(action_bet.bet)):
                    assert(floatIsEqual(action_bet.bet[k],genBet.uniform(0,1)))
                for k in range(len(action_bet.prediction)):
                    assert(
                        floatIsEqual(
                            action_bet.prediction[k], 
                            genPred.uniform(
                                votingConf.min_possible_vote_total(),
                                votingConf.max_possible_vote_total()
                            )
                        )
                    )

@pytest.mark.parametrize("N", [
    1,
    2,
    5,
    20,
    15
])
def test_random_simple_agent_forward_prediction(N):
    vote, seed = (5,0)
    print(f"random: {vote} , {seed}")
    agent_fac = factory.AgentFactory()
    env_factory = factory.EnvFactory()
    votingConfigFac = factory.VotingConfigFactory()
    votingConf = votingConfigFac.create({})
    agent = agent_fac.create({
        'type':'random',
        'vote':vote,
        'N': N,
        'seed': seed,
        'totalVotesBound': (
            votingConf.min_possible_vote_total,
            votingConf.max_possible_vote_total)
        })
    env = env_factory.create({})
    genBet = np.random.default_rng(seed=seed)
    genPred = np.random.default_rng(seed=seed)
    for _ in range(100):
        assert(agent.vote(state) == vote)
        for j in env.actions():
            state = env.state()
            action_bet = agent.bet(state,j,1)
            assert(
                len(action_bet.bet) == N
            )
            assert(
                len(action_bet.prediction) == N
            )
            for k in range(len(action_bet.bet)):
                assert(floatIsEqual(action_bet.bet[k],genBet.uniform(0,1)))
            for k in range(len(action_bet.prediction)):
                assert(
                    floatIsEqual(
                        action_bet.prediction[k], 
                        genPred.uniform(
                            votingConf.min_possible_vote_total(k),
                            votingConf.max_possible_vote_total(k)
                        )
                    )
                )


############ NOT ALLOWED ############
# @pytest.mark.parametrize("config",#
#     random_agent_config           #
# )                                 #
# def test_checking(config):        #
#     print(config)                 #
#####################################