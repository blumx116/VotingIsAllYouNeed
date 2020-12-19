# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-06 17:15:14


from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Callable, Dict, Optional

from VIAYN.project_types import VotingConfiguration, VoteRange
from VIAYN.samples.voting import (
    ClassicalVotingConfig, DirectCardinalVotingConfig,
    RecommendedVotingConfig)

@unique
class VotingConfigEnum(Enum):
    simple = auto()
    suggested = auto()
    classical = auto()


@dataclass(frozen=True)
class VotingConfigFactorySpec:
    """
    Typed Spec to input to [VotingConfigFactory.create] to create an [VotingConfig]
    
    Note: Typed spec was used instead of Dict for the factory to provide
        more information to users.

    Parameters
    ----------
    configType: VotingConfigEnum:
        type of Voting Config corresponds to simple and suggested in requirements
        classical is a straight sum of binary votes
    voteRange: VoteRange
        Provides a range of values for agent votes
    n_agents: Optional[int] = 0
        set the initial number of agents for the voting configuration
    """
    configType: VotingConfigEnum
    voteRange: VoteRange
    n_agents: Optional[int] = 0
    
    def __post_init__(self):
        assert(self.voteRange is not None)


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
    lookup: Dict[VotingConfigEnum, Callable[[VoteRange], VotingConfiguration]] = {
        VotingConfigEnum.simple: lambda vr: DirectCardinalVotingConfig(vr),
        VotingConfigEnum.suggested: lambda vr: RecommendedVotingConfig(vr),
        VotingConfigEnum.classical: lambda vr: ClassicalVotingConfig(vr)
    }

    @staticmethod
    def create(spec: VotingConfigFactorySpec) -> VotingConfiguration:
        return VotingConfigFactory.lookup[spec.configType](spec.voteRange)
