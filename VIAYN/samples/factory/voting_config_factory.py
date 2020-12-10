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
    lookup: Dict[VotingConfigEnum, Callable[[], VotingConfiguration]] = {
        VotingConfigEnum.simple: lambda: DirectCardinalVotingConfig(),
        VotingConfigEnum.suggested: lambda: RecommendedVotingConfig(),
        VotingConfigEnum.classical: lambda: ClassicalVotingConfig()
    }

    @staticmethod
    def create(spec: VotingConfigFactorySpec) -> VotingConfiguration:
        return VotingConfigFactory.lookup[spec.configType]()
