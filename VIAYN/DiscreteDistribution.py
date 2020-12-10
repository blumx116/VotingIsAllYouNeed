from dataclasses import dataclass
from typing import List, Optional, Dict, Sequence, Iterable
from copy import copy
from typing import Dict, List, Sequence, TypeVar

from VIAYN.project_types import A, S, WeightedBet

import numpy as np
from numpy.random import RandomState

from VIAYN.project_types import A, S, WeightedBet

_epsilon_: float = 1e-4


@dataclass(frozen=True)
class DiscreteDistribution:
    values: List[float]
    probabilities: List[float]
    random_seed: Optional[np.random] = None

    def __post_init__(self):
        assert len(np.unique(self.values)) == len(self.values)
        assert abs(np.sum(self.probabilities) - 1.) < _epsilon_
        assert len(self.values) == len(self.probabilities)

    def get_random(self):
        return self.random_seed if self.random_seed is not None else RandomState()

    def sample(self) -> float:
        random_value: float = self.get_random().uniform()
        proba_sum: float = 0.
        val: float
        proba: float
        for val, proba in zip(self.values, self.probabilities):
            proba_sum += proba
            if proba_sum >= random_value:
                return val
        assert False, "Should never get here"

    @staticmethod
    def from_weighted_vals(
            vals: Iterable[float],
            weights: Iterable[float],
            random_seed: RandomState) -> "DiscreteDistribution":
        distribution_constructor: Dict[float, float] = {}
        val: float
        weight: float
        for val, weight in zip(vals, weights):
            if val not in distribution_constructor:
                distribution_constructor[val] = 0.
            distribution_constructor[val] += weight

        total_weights: float = sum(distribution_constructor.values())

        return DiscreteDistribution(
            values=list(distribution_constructor.keys()),
            probabilities=[weight / total_weights for weight in distribution_constructor.values()],
            random_seed=random_seed)
