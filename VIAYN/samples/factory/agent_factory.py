# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:42:44
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-06 15:36:25

from enum import Enum, unique, auto
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Callable, Dict, Iterable
import numpy as np
from VIAYN.project_types import Agent, VoteBoundGetter
from VIAYN.samples.agents import (
    BetSelectionMechanism,
    PredictionSelectionMechanism,
    VotingMechanism,
    CompositeBettingMechanism,
    CompositeAgent,
    StaticVotingMechanism,
    StaticBetSelectionMech,
    StaticPredSelectionMech,
    RNGUniforPredSelectionMech
)

@unique
class AgentsEnum(Enum):
    # constant agent returns the same votes, bets and predictions that are specified
    constant = auto()
    # random agent returns random predictions that are specified with length N
    random = auto()


@dataclass(frozen=True)
class AgentFactorySpec:
    """
    Typed Spec to input to [AgentFactory.create] to create an [Agent]
    
    Note: Typed spec was used instead of Dict for the factory to provide
        more information to users.

    Parameters
    ----------
    agentType: AgentsEnum
        which agent to create possible types are (constant/random) currently
    vote: float
        what value should the agent's happiness be. Agent's happiness is constant
        for now
    totalVotesBound: Optional[Tuple[VoteBoundGetter, VoteBoundGetter]] = None
        What predictions values can be predicted by the agent (min, max)
    seed: Optional[int]
        Random seed used by random processes of agents
    prediction: Optional[Union[float, List[float]]] = None
        what agents predict the total happiness of all agents at each timestep.
        If prediction is a List of floats, you would be predicting the happiness
        for later in the future. 
        e.g. [5.5,10.7,30.2] at time-step 4 corresponds to the total happiness of the 
        system is 10.7 for all agents.
        money wil be bet at time-step 5.
    bet: Optional[Union[float, List[float]]] = None
        what percent of the money an agent predicts at each timestep.
        If bet is a List of floats, you would be casting a bet for later in the
        future. e.g. [0,0.5,0.2] at time-step 4 corresponds to 0.5 of an agents
        money wil be bet at time-step 5.  
    N: Optional[int] = None
        length of bets as specified by requirements
    """
    agentType: AgentsEnum
    vote: float
    totalVotesBound: Optional[Tuple[VoteBoundGetter, VoteBoundGetter]] = None
    seed: Optional[int] = None
    prediction: Optional[Union[float, List[float]]] = None
    bet: Optional[Union[float, List[float]]] = None
    N: Optional[int] = None
    
    def __post_init__(self):
        # constant agent uses these params in addition to vote at least
        if self.agentType == AgentsEnum.constant:
            assert(self.bet is not None)
            assert(self.prediction is not None)
        # random agent uses these params in addition to vote at least
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
        assert spec.bet is not None
        return StaticBetSelectionMech(
            AgentFactory._repeat_if_float_(spec.bet, spec.N))

    @staticmethod
    def _create_static_vote_selection_(spec: AgentFactorySpec) -> VotingMechanism:
        return StaticVotingMechanism(spec.vote)

    @staticmethod
    def _create_static_prediction_selection_(spec: AgentFactorySpec) -> PredictionSelectionMechanism:
        assert spec.prediction is not None
        return StaticPredSelectionMech(
            AgentFactory._repeat_if_float_(spec.prediction, spec.N))

    @staticmethod
    def _create_rng_uniform_prediction_selection_(spec: AgentFactorySpec) -> PredictionSelectionMechanism:
        assert spec.totalVotesBound is not None
        assert spec.seed is not None
        assert spec.N is not None
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
    def _repeat_if_float_(
            value: Union[float, List[float]],
            n: Optional[int] = None) -> List[float]:
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
            assert 0 <= value <= 1
            res: List[float] = [value*1.0/n] * n
            floatErr = abs(1-np.sum(res))
            for i in range(len(res)):
                res[i] = max(0,res[i] - floatErr)
                floatErr -= res[i]
                if (floatErr <= 0):
                    return res
            return res
        else:
            assert isinstance(value, Iterable)
            if n is not None:
                assert len(value) == n
            return value

    _creators_: Dict[AgentsEnum, Callable[[AgentFactorySpec], Agent]] = {
        AgentsEnum.random: lambda x: AgentFactory._create_random_agent_(x),
        AgentsEnum.constant: lambda x: AgentFactory._create_static_agent_(x)
    }
