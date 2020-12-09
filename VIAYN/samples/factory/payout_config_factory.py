# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 17:09:12


from dataclasses import dataclass
from enum import Enum, unique, auto
from VIAYN.project_types import PayoutConfiguration


@unique
class PayoutConfigEnum(Enum):
    simple = auto()
    suggested = auto()


@dataclass(frozen=True)
class PayoutConfigFactorySpec:
    configType: PayoutConfigEnum

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


