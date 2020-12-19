# -*- coding: utf-8 -*-
# @Author: Carter.Blum, Suhail.Alnahari
# @Date:   2020-12-05 16:54:09
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 17:09:12
from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Dict, Callable, List

from VIAYN.project_types import PayoutConfiguration, Weighted
from VIAYN.samples.payout import SuggestedPayoutConfig, SimplePayoutConfig
from VIAYN.samples.payout import PayoutConfigBase
from VIAYN.utils import weighted_quartile


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

UBType = Callable[[List[Weighted]], float]


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
    lookup: Dict[PayoutConfigEnum, Callable[[UBType], PayoutConfiguration]] = {
        PayoutConfigEnum.simple: lambda ub_fn: SimplePayoutConfig(ub_fn),
        PayoutConfigEnum.suggested: lambda ub_fn: SuggestedPayoutConfig(ub_fn)
    }

    upper_bound_lookup: Dict[UpperBoundConfigEnum, UBType] = {
        UpperBoundConfigEnum.max: PayoutConfigBase.max_loss,
        UpperBoundConfigEnum.quartile95: lambda weights: weighted_quartile(weights, 0.95)
    }

    @staticmethod
    def create(spec: PayoutConfigFactorySpec) -> PayoutConfiguration:
        ub_fn: UBType = PayoutConfigFactory.upper_bound_lookup[spec.upperBound]
        return PayoutConfigFactory.lookup[spec.configType](ub_fn)
