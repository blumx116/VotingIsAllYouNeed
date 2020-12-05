# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 20:23:15
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-04 21:21:45

from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, List, Callable, Dict,TypeVar, Generic
from enum import Enum, unique, auto

from VIAYN.project_types import (
    Agent,
    Environment,
    PayoutConfiguration,
    VotingConfiguration,
    PolicyConfiguration)
from VIAYN.samples.agents import (
    BetSelectionMechanism, PredictionSelectionMechanism, VotingMechanism, BettingMechanism, CompositeBettingMechanism,
    CompositeAgent, StaticVotingMechanism, StaticBetSelectionMech, StaticPredSelectionMech, StaticBettingMechanism,
    RNGUniforPredSelectionMech
)

VoteBoundGetter = Callable[[int], float]

@unique
class AgentsEnum(Enum):
    constant = auto()
    random = auto()

VoteBoundGetter = Callable[[int], float]
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
        if (self.agentType == AgentsEnum.constant):
            assert(self.bet is not None)
            assert(self.prediction is not None)
        elif(self.agentType == AgentsEnum.random):
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
    def create(spec: AgentFactorySpec) -> Agent:
        assert spec.agentType in AgentFactory._creators_
        return AgentFactory._creators_[spec.agentType](spec)

    @staticmethod
    def _create_static_agent_(
            spec: AgentFactorySpec) -> Agent:
        return CompositeAgent(
            voting_mechanism=AgentFactory._create_static_vote_selection_(spec),
            betting_mechanism=CompositeBettingMechanism(
                prediction_selection=AgentFactory._create_static_prediction_selection_(spec),
                bet_selection=AgentFactory._create_static_bet_selection_(spec)))

    @staticmethod
    def _create_static_bet_selection_(
            spec: AgentFactorySpec) -> BetSelectionMechanism:
        return StaticBetSelectionMech(
            AgentFactory._repeat_if_float_(spec.bet, spec.N))

    @staticmethod
    def _create_static_vote_selection_(spec: AgentFactorySpec) -> VotingMechanism:
        return StaticVotingMechanism(spec.vote)

    @staticmethod
    def _create_static_prediction_selection_(spec: AgentFactorySpec) -> PredictionSelectionMechanism:
        return StaticPredSelectionMech(
            AgentFactory._repeat_if_float_(spec.prediction, spec.N))

    @staticmethod
    def _create_rng_uniform_prediction_selection_(spec: AgentFactorySpec) -> PredictionSelectionMechanism:
        assert spec.totalVotesBound is not None
        assert spec.seed is not None
        return RNGUniforPredSelectionMech(
            tsteps_per_prediction=spec.N,
            min_possible_prediction=spec.totalVotesBound[0],
            max_possible_prediction=spec.totalVotesBound[1],
            random_seed=spec.seed)

    @staticmethod
    def _create_random_agent_(
            spec: AgentFactorySpec) -> Agent:
        return CompositeAgent(
            voting_mechanism=AgentFactory._create_static_vote_selection_(spec),
            betting_mechanism=CompositeBettingMechanism(
                prediction_selection=AgentFactory._create_rng_uniform_prediction_selection_(spec),
                bet_selection=AgentFactory._create_static_bet_selection_(spec)))

    @staticmethod
    def _repeat_if_float_(self,
            value: Union[float, List[float]],
            n: Optional[int] = None):
        """
        Utility method.
        If a float is passed in for 'value', returns [value] * N
        otherwise, just returns value, assuming it is a list
        :param n: int
            number of timesteps per prediction
            Only necessary if value is a float.
            Otherwise, just checks that 'value' has length N
        :param value:
            prediction or bet. If float, same value used fro all timesteps
        :return: values: List[float]
            values as a list, if conversion is necessary
        """
        if isinstance(value, float):
            # repeat for each timestep
            assert n is not None
            return [value] * n
        else:
            assert isinstance(value, Iterable)
            if n is not None:
                assert len(value) == n
            return value

    _creators_: Dict[str, Callable[[AgentFactorySpec], Agent]] = {
        'random': _create_random_agent_,
        'static': _create_static_agent_
    }

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