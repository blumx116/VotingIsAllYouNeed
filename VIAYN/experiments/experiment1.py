from typing import Any, TypeVar
from VIAYN.project_types import (
    A, Agent, Environment, 
    PolicyConfiguration, VotingConfiguration, PayoutConfiguration,
    SystemConfiguration)
from VIAYN.samples.factory import (
    PolicyConfigEnum, PolicyConfigFactory, PolicyConfigFactorySpec,
    PayoutConfigEnum, PayoutConfigFactory, PayoutConfigFactorySpec,
    VotingConfigEnum, VotingConfigFactory, VotingConfigFactorySpec,
    AgentFactory, AgentFactorySpec, AgentsEnum,
    EnvFactory, EnvsEnum, EnvsFactorySpec)
from VIAYN.samples.vote_ranges import BinaryVoteRange
from VIAYN.samples.env import IntAction as IA
from VIAYN.train import train


agent1: Agent[IA, IA] = AgentFactory.create(
    AgentFactorySpec(AgentsEnum.constant, 
    vote=1.,
    totalVotesBound=(lambda _: 0, lambda _: 2.), 
    prediction=1.,
    bet=0.5,
    N=1))

agent2: Agent[IA, IA] = AgentFactory.create(
    AgentFactorySpec(AgentsEnum.random,
    vote=0.,
    totalVotesBound=(lambda _: 0, lambda _: 2.),
    seed=0,
    bet=0.5,
    N=1))

environment: Environment[IA, IA] = EnvFactory.create(
    EnvsFactorySpec(EnvsEnum.default, n_actions=2))

vc: VotingConfiguration[IA, IA] = VotingConfigFactory.create(
    VotingConfigFactorySpec(
        configType=VotingConfigEnum.suggested,
        voteRange=BinaryVoteRange(),
        n_agents=2))

poc: PolicyConfiguration[IA, float, IA] = PolicyConfigFactory.create(
    PolicyConfigFactorySpec(
        PolicyConfigEnum.simple))

pac: PayoutConfiguration[IA, IA] = PayoutConfigFactory.create(
    PayoutConfigFactorySpec(
        PayoutConfigEnum.simple))

config: SystemConfiguration[IA, float, IA] = SystemConfiguration(
    vc, poc, pac)

result: Any = train([agent1, agent2], environment, range(10), config, tsteps_per_episode=10)

print('end')