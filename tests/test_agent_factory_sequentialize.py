import pytest

from VIAYN.samples.factory.agent_factory import AgentFactory, AgentFactorySpec, AgentsEnum
from VIAYN.samples.factory.env_factory import EnvsEnum, EnvFactory, EnvsFactorySpec
from VIAYN.project_types import AnonymizedHistoryItem
from tests.conftest import sequenceEqual, floatIsEqual


def check_same_behaviour(agent1, agent2, state, action, money=1.):
    for _ in range(5):
        assert floatIsEqual(agent1.vote(state), agent2.vote(state)),\
            "Should be able to call vote any number of times without changing"
        b1 = agent1.bet(state, action, money)
        b2 = agent2.bet(state, action, money)
        assert sequenceEqual(b1.bet, b2.bet), \
            "Should be able to call bet any number of times without changing"
        assert sequenceEqual(b1.prediction, b2.prediction), \
            "Should be able to call bet any number of times without changing"


def test_base_case():
    """
    Simple base case test.
    Creates static agents, put s them in a sequence, and checks that it gives the
    correct output at each timestep.
    Note that vote() and bet() are both called multiple times per timestep
    """
    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action

    spec1 = AgentFactorySpec(AgentsEnum.constant, vote=3., prediction=[1, 2], bet=[.1, .2])
    spec2 = AgentFactorySpec(AgentsEnum.constant, vote=4., prediction=[2, 2], bet=[.1, .0])
    spec3 = AgentFactorySpec(AgentsEnum.constant, vote=2., prediction=[3, 1], bet=[.3, .1])

    agent1 = AgentFactory.create(spec1)
    agent2 = AgentFactory.create(spec2)
    agent3 = AgentFactory.create(spec3)

    comp_agent = AgentFactory.sequentialize([agent1, agent2, agent3], [2, 3, 4])
    delegates = [agent1, agent1, agent2, agent2, agent2, agent3, agent3]

    for i in range(7):
        check_same_behaviour(comp_agent, delegates[i], state, action)
        comp_agent.view(AnonymizedHistoryItem())

def test_stateful_agents():
    """
    In the previous test all of the agents were static - e.g. calling them twice should give the same results
    Here, we replace them with random agents with a specified seed so that we can identify if they are giving the
    correct results.
    """
    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action

    vote_bounds = (lambda x: 0., lambda x: 2.)

    spec1 = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2)
    spec2 = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=2, N=2)

    agent1 = AgentFactory.create(spec1)
    agent2 = AgentFactory.create(spec2)
    agent1_duplicate = AgentFactory.create(spec1)
    agent2_duplicate = AgentFactory.create(spec2)

    comp_agent = AgentFactory.sequentialize([agent1_duplicate, agent2_duplicate], [3, 3])
    delegates = [agent1, agent1, agent1, agent2, agent2]

    for i in range(5):
        check_same_behaviour(comp_agent, delegates[i], state, action)
        comp_agent.view(AnonymizedHistoryItem())
        delegates[i].view(AnonymizedHistoryItem())


def test_unequal_lengths():
    """
    This is mostly just input checks.
    The number of agents should be 1 more than the number of transitions
    We make a number of calls where this relationship is violated and expect them to error.
    """
    vote_bounds = (lambda x: 0., lambda x: 3.)

    spec1 = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2)
    spec2 = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=2, N=2)
    spec3 = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=3, N=2)
    spec4 = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=4, N=2)

    agent1 = AgentFactory.create(spec1)
    agent2 = AgentFactory.create(spec2)
    agent3 = AgentFactory.create(spec3)
    agent4 = AgentFactory.create(spec4)

    with pytest.raises(Exception):
        AgentFactory.sequentialize([agent1, agent2], [1,2,3,4,5])
        # too many transitions

    with pytest.raises(Exception):
        AgentFactory.sequentialize([agent1, agent2, agent3], [1, 2])
        # too many agents

    with pytest.raises(Exception):
        AgentFactory.sequentialize([agent1, agent2, agent4], [1, 2, 3, 2])


def test_empty_arguments():
    """
    Tests the case where at least one of agents or transitions has length 0 when being passed to
    sequentialize()
    """
    vote_bounds = (lambda x: 0., lambda x: 3.)
    spec = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2)
    agent = AgentFactory.create(spec)

    with pytest.raises(Exception):
        AgentFactory.sequentialize([agent], [])
        # it's possible this shouldn't raise an excpetion

    with pytest.raises(Exception):
        AgentFactory.sequentialize([], [1,])
        # redundant with checking that the list is too long

    with pytest.raises(Exception):
        AgentFactory.sequentialize([], [])
        # also redunndant


def test_super_long():
    """
    Literally just a test with a bajillion agents.
    Each agent lasts for 3 timesteps.
    """
    N_AGENTS: int = 10000
    FREQ: int = 3

    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action
    vote_bounds = (lambda x: 0., lambda x: 3.)
    spec = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2)

    agents = [AgentFactory.create(spec) for _ in range(N_AGENTS)]
    duplicate_agents = [AgentFactory.create(spec) for _ in range(N_AGENTS)]
    transitions = [FREQ] * N_AGENTS

    composite = AgentFactory.sequentialize(duplicate_agents, switch_at=transitions)

    for i in range(FREQ * N_AGENTS):
        delegated = agents[i // FREQ]
        check_same_behaviour(composite, delegated, state, action)
        composite.view(AnonymizedHistoryItem())
        delegated.view(AnonymizedHistoryItem())

def test_duplicate_agents():
    """
    Similar to previous tests, except that the same agent comes up multiple times in the agents list
    TODO: test the case where agent functionality of the inner agent changes whenever View() is called
    """
    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action
    vote_bounds = (lambda x: 0., lambda x: 3.)
    spec = AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2)
    agent = AgentFactory.create(spec)
    duplicate_agent = AgentFactory.create(spec)

    composite_agent = AgentFactory.sequentialize([agent] * 10, [2] * 10)

    for _ in range(20):
        check_same_behaviour(composite_agent, duplicate_agent, state, action)
        composite_agent.view(AnonymizedHistoryItem())
        duplicate_agent.view(AnonymizedHistoryItem())


def test_recursive_case():
    """
    Tests the case where some of the agents provided to sequentialize are themselves the result of sequentialize.
    It's pretty likely that View() isn't being forwarded properly if this one doesn't work.
    """
    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action
    vote_bounds = (lambda x: 0., lambda x: 3.)

    specs = [
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=2, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=3, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=4, N=2)]

    agents = [AgentFactory.create(spec) for spec in specs]
    duplicate_agents = [AgentFactory.create(spec) for spec in specs]

    comp1 = AgentFactory.sequentialize([duplicate_agents[0], duplicate_agents[1]], [3, 3])
    comp2 = AgentFactory.sequentialize([duplicate_agents[2], duplicate_agents[3]], [3, 3])
    big_comp = AgentFactory.sequentialize([comp1, comp2, duplicate_agents[0]], [5, 4, 2])

    delegates = ([agents[0]] * 3) + ([agents[1]] * 2) + ([agents[2]] * 3) + \
                ([agents[3]] * 1) + ([agents[0]] * 2)

    for i in range(10):
        check_same_behaviour(big_comp, delegates[i], state, action)
        big_comp.view(AnonymizedHistoryItem())
        delegates[i].view(AnonymizedHistoryItem())

def test_many_timesteps():
    """
    Similar to above tests, except that we just run for an extra 1000 timesteps
    after the last transition to make sure that it keeps using the last agent.
    Returns
    -------

    """
    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action
    vote_bounds = (lambda x: 0., lambda x: 3.)

    specs = [
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=2, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=3, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=4, N=2)]

    agents = [AgentFactory.create(spec) for spec in specs]
    duplicate_agents = [AgentFactory.create(spec) for spec in specs]

    big_comp = AgentFactory.sequentialize(duplicate_agents, [2, 3, 4, 1])

    delegates = ([agents[0]] * 2) + ([agents[1]] * 3) + ([agents[2]] * 4) + \
                ([agents[3]] * 1)

    for i in range(10):
        check_same_behaviour(big_comp, delegates[i % len(delegates)], state, action)
        big_comp.view(AnonymizedHistoryItem())
        delegates[i].view(AnonymizedHistoryItem())


def test_negative_duration():
    env = EnvFactory.create(EnvsFactorySpec(EnvsEnum.default, n_actions=3))
    state = env.state()
    action = env.last_action
    vote_bounds = (lambda x: 0., lambda x: 3.)

    specs = [
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=1., seed=1,N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=2, N=2),
        AgentFactorySpec(AgentsEnum.random, bet=0.5, totalVotesBound=vote_bounds, vote=0., seed=3, N=2)]

    agents = [AgentFactory.create(spec) for spec in specs]

    with pytest.raises(Exception):
        AgentFactory.sequentialize(agents, [10, -1])
        # shouldn't be able to have a negative duration
