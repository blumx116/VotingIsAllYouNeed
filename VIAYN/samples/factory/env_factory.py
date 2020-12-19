# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:51:55
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 16:52:57

from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Callable, Dict, Optional

from VIAYN.project_types import Environment
from VIAYN.samples.env import StaticEnvironment


@unique
class EnvsEnum(Enum):
    default = auto()


@dataclass(frozen=True)
class EnvsFactorySpec:
    envType: EnvsEnum
    n_actions: Optional[int] = None # number is exact at each state

    def __post_init__(self):
        if self.envType ==  EnvsEnum.default:
            assert self.n_actions is not None
        else:
            assert False, "default is currently only supported type"


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
        if spec.envType == EnvsEnum.default:
            assert spec.n_actions is not None
            return StaticEnvironment(spec.n_actions)
        assert False, "Can only create static environment right now"
