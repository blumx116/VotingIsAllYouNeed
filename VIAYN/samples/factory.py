# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 20:23:15
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-04 20:13:15

from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, List, Callable, Dict,TypeVar, Generic
from enum import Enum, unique, auto

from VIAYN.project_types import (
    Agent,
    Environment,
    PayoutConfiguration,
    VotingConfiguration,
    PolicyConfiguration
)

VoteBoundGetter = Callable[[int], float]


@unique
class AgentsEnum(Enum):
    constant = auto()
    random = auto()


@dataclass(frozen=True)
class AgentFactorySpec:
    agentType: AgentsEnum
    vote: float
    totalVotesBound: Optional[Tuple[VoteBoundGetter, VoteBoundGetter]] = None
    seed: Optional[float] = None
    prediction: Optional[Union[float, List[float]]] = None
    bet: Optional[Union[float, List[float]]] = None
    N: Optional[int] = None
    
    def __post_init__(self):
        if self.agentType == AgentsEnum.constant:
            assert(self.bet is not None)
            assert(self.prediction is not None)
        elif self.agentType == AgentsEnum.random:
            assert(self.N is not None)
            assert(self.totalVotesBound is not None)
            assert(self.bet is not None)
        else:
            raise TypeError(self.agentType)


class AgentFactory:
    """
    Creates different types of Agents Based on spec

    Parameters
    ----------
    spec: AgentFactorySpec
        Specifications to create agent with, see above

    Returns
    -------
    Agent
        created agent based on spec
    """
    @staticmethod
    def create(self, spec: AgentFactorySpec) -> Agent:
        ...


@unique
class EnvsEnum(Enum):
    default = auto()


@dataclass(frozen=True)
class EnvsFactorySpec:
    envType: EnvsEnum

    def __post_init__(self):
        assert(self.envType == EnvsEnum.default)


class EnvFactory:
    """
    Creates different types of Environments Based on spec


    Parameters
    ----------
    spec: EnvsFactorySpec
        Specifications to create Environment with, see above

    Returns
    -------
    Environment
        created environment based on spec
    """
    @staticmethod
    def create(spec: EnvsFactorySpec) -> Environment:
        ...


@dataclass(frozen=True)
class PayoutConfigFactorySpec:
    def __post_init__(self):
        pass


class PayoutConfigFactory:
    """
    Creates different types of Payout Configs Based on spec


    Parameters
    ----------
    spec: PayoutConfigFactorySpec
        Information to initialize Payout Config

    Returns
    -------
    PayoutConfiguration
        created payout config based on spec
    """
    @staticmethod
    def create(spec: PayoutConfigFactorySpec) -> PayoutConfiguration:
        ...


@dataclass(frozen=True)
class PolicyConfigFactorySpec:
    def __post_init__(self):
        pass


class PolicyConfigFactory:
    """
    Creates different types of Policy Configs Based on spec

    List of acceptable configs:
    TBD

    Parameters
    ----------
    spec: PolicyConfigFactorySpec
        Information to initialize Policy Config

    Returns
    -------
    PolicyConfiguration
        created policy config based on spec
    """
    @staticmethod
    def create(spec: PolicyConfigFactorySpec) -> PolicyConfiguration:
        ...


@dataclass(frozen=True)
class VotingConfigFactorySpec:
    def __post_init__(self):
        pass


class VotingConfigFactory:
    """
    Creates different types of Voting Configs Based on spec

    
    Parameters
    ----------
    spec: VotingConfigFactorySpec
        Information to initialize Voting Config

    Returns
    -------
    VotingConfiguration
        created voting config based on spec
    """
    @staticmethod
    def create(spec: VotingConfigFactorySpec) -> VotingConfiguration:
        ...
