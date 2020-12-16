from VIAYN.project_types import Agent
from VIAYN.samples.factory import (
    PolicyConfigEnum, PolicyConfigFactory, PolicyConfigFactorySpec,
    PayoutConfigEnum, PayoutConfigFactory, PayoutConfigFactorySpec,
    VotingConfigEnum, VotingConfigFactory, VotingConfigFactorySpec,
    AgentFactory, AgentFactorySpec, AgentsEnum,
    EnvFactory, EnvsEnum, EnvsFactorySpec)

from VIAYN.samples.env import IntAction

agent1: Agent[IntAction, IntAction] = AgentFactory.create(
    AgentFactorySpec(AgentsEnum.constant, 
    vote=1.,
    totalVotesBound=(0, 2), 
    prediction=1.,
    bet=0.5,
    N=2))