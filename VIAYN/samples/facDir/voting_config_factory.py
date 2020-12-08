# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 17:08:15


from dataclasses import dataclass
from enum import Enum, unique, auto
from VIAYN.project_types import VotingConfiguration


@unique
class VotingConfigEnum(Enum):
    simple = auto()
    suggested = auto()


@dataclass(frozen=True)
class VotingConfigFactorySpec:
    configType: VotingConfigEnum

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
