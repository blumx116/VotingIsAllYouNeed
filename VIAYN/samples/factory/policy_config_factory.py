# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 17:07:32


from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Dict, Callable

from VIAYN.project_types import PolicyConfiguration
from VIAYN.samples.policy import (
    GreedyPolicyConfiguration,
    ThompsonPolicyConfiguration,
    ThompsonPolicyConfiguration2)


@unique
class PolicyConfigEnum(Enum):
    simple = auto()
    suggested = auto()
    suggested_general = auto()


@dataclass(frozen=True)
class PolicyConfigFactorySpec:
    configType: PolicyConfigEnum

    def __post_init__(self):
        pass


class PolicyConfigFactory:
    """
    Creates different types of Policy Configs Based on spec


    Parameters
    ----------
    spec: PolicyConfigFactorySpec
        Information to initialize Policy Config

    Returns
    -------
    PolicyConfiguration
        created policy config based on spec
    """
    lookup: Dict[PolicyConfigEnum, Callable[[], PolicyConfiguration]] = {
        PolicyConfigEnum.simple: lambda: GreedyPolicyConfiguration(),
        PolicyConfigEnum.suggested: lambda: ThompsonPolicyConfiguration2(),
        PolicyConfigEnum.suggested_general: lambda: ThompsonPolicyConfiguration()
    }

    @staticmethod
    def create(spec: PolicyConfigFactorySpec) -> PolicyConfiguration:
        return PolicyConfigFactory.lookup[spec.configType]()


