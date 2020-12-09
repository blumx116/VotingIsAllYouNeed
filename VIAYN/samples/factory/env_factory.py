# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:51:55
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 16:52:57

from dataclasses import dataclass
from enum import Enum, unique, auto
from VIAYN.project_types import Environment


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