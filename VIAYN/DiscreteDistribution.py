# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-11 20:45:25
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-11 20:45:25
from dataclasses import dataclass
from typing import List, Optional, Dict, Sequence, Iterable
from copy import copy
from typing import Dict, List, Sequence, TypeVar

from VIAYN.project_types import A, S, WeightedBet

import numpy as np
from numpy.random import Generator, default_rng

from VIAYN.project_types import A, S, WeightedBet

_epsilon_: float = 1e-4


@dataclass(frozen=True)
class DiscreteDistribution:
    """
    Data class representing a list of real-valued possibilities
    associated with probabilities, such that the total probability is always 1
    """
    values: List[float]
    probabilities: List[float]
    random_seed: Optional[Generator] = None

    def __post_init__(self):
        assert len(np.unique(self.values)) == len(self.values)
        assert abs(np.sum(self.probabilities) - 1.) < _epsilon_
        assert len(self.values) == len(self.probabilities)

    def get_random(self):
        return self.random_seed if self.random_seed is not None else default_rng()

    def sample(self) -> float:
        """
        Generates a random sample from the distribution in O(n) time,
        where N is the number of elements in the distribution

        Essentially generates a random number between 0 & 1 and then
        selects the value corresponding to that number in the CDF
        In practice, sorting the numbers to obtain a real CDF is useless,
        so we don't use an actual CDF

        TODO: use numpy for this instead, it's likely faster

        Returns
        -------
        sample_val: float
            randomly selected value form the distribution
        """

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
            random_seed: Generator) -> "DiscreteDistribution":
        """
        Creates a new DiscreteDistribution from a series of values & their weights.
        The probability of a value in the distribution is proportional to its weight.
        If a value occurs more than once, its corresponding weights are summed

        Weights are automatically normalized

        Parameters
        ----------
        vals: Iterable[float]
            the values in the discrete distribution
        weights: Iterable[float]
            the weights of each of the values in vals.
            Should correspond to each of the vals
        random_seed: Generator
            the random seed to be used when sampling from the returned object

        Returns
        -------
        distibution: DiscreteDistribution
            the newly constructed DiscreteDistribution
        """
        distribution_constructor: Dict[float, float] = {}
        val: float
        weight: float
        for val, weight in zip(vals, weights):
            if val not in distribution_constructor:
                distribution_constructor[val] = 0.
            distribution_constructor[val] += weight
        # construct a dictionary mapping values to weights

        total_weights: float = sum(distribution_constructor.values())
        # calculate total weights for normalization

        return DiscreteDistribution(
            values=list(distribution_constructor.keys()),
            probabilities=[weight / total_weights for weight in distribution_constructor.values()],
            random_seed=random_seed)
