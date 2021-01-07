# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 17:07:32


from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Dict, Callable, Optional

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
    """
    Typed Spec to input to [PolicyConfigFactory.create] to create an [PolicyConfig]
    
    Note: Typed spec was used instead of Dict for the factory to provide
        more information to users.

    Parameters
    ----------
    configType: PolicyConfigEnum
        type of Policy Config corresponds to simple and suggested in requirements
        suggested general samples each action and timestep using Thompson while
        suggested samples each action after summing across time-steps
    random_seed: Optional[int]
        random seed, only used in ThompsonPolicies at the moment
    """
    configType: PolicyConfigEnum
    random_seed: Optional[int]

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
    lookup: Dict[PolicyConfigEnum, Callable[[Optional[int]], PolicyConfiguration]] = {
        PolicyConfigEnum.simple: lambda seed: GreedyPolicyConfiguration(),
        PolicyConfigEnum.suggested: lambda seed: ThompsonPolicyConfiguration2(seed),
        PolicyConfigEnum.suggested_general: lambda seed: ThompsonPolicyConfiguration(seed)
    }

    @staticmethod
    def create(spec: PolicyConfigFactorySpec) -> PolicyConfiguration:
        return PolicyConfigFactory.lookup[spec.configType](spec.random_seed)
