# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 17:09:12
from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Dict, Callable

from VIAYN.project_types import PayoutConfiguration
from VIAYN.samples.payout import SuggestedPayoutConfig, SimplePayoutConfig


@unique
class PayoutConfigEnum(Enum):
    simple = auto()
    suggested = auto()

@unique
class UpperBoundConfigEnum(Enum):
    max = auto()
    quartile95 = auto()


@dataclass(frozen=True)
class PayoutConfigFactorySpec:
    configType: PayoutConfigEnum
    upperBound: UpperBoundConfigEnum = UpperBoundConfigEnum.max

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
    lookup: Dict[PayoutConfigEnum, Callable[[], PayoutConfiguration]] = {
        PayoutConfigEnum.simple: lambda: SimplePayoutConfig(),
        PayoutConfigEnum.suggested: lambda: SuggestedPayoutConfig()
    }

    @staticmethod
    def create(spec: PayoutConfigFactorySpec) -> PayoutConfiguration:
        return PayoutConfigFactory.lookup[spec.configType]()
