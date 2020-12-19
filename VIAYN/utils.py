# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-12-10 14:54:22
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:59:06
from copy import copy
from typing import Dict, List, Sequence, TypeVar, Optional, Callable, Tuple, Iterable

import numpy as np

from VIAYN.project_types import A, S, WeightedBet, Weighted

T = TypeVar("T")
V = TypeVar("V", int, float, complex, str)


def add_dictionaries(
        first: Dict[T, V],
        second: Dict[T, V]) -> Dict[T, V]:
    """
    Combines two dictionaries. If a key is found in both dictionaries,
        the values are added
    Both arguments are not modified
    V must implement __add__(V, V) -> V
    Parameters
    ----------
        first: Dict[T, V]
        second: Dict[T, V]
    Returns
    -------
        combined : Dict[T, V]
    """
    result: Dict[T, V] = copy(first)
    key: T
    value: V
    for key, value in second.items():
        if key in result:
            result[key] += value
        else:
            result[key] = value
    return result


def weighted_mean_of_bets(bets: List[WeightedBet[A, S]]) -> List[float]:
    # TODO: should probably use np.average with weights
    assert len(bets) > 0
    assert len(np.unique([len(bet.bet) for bet in bets])) == 1
    assert len(np.unique([len(bet.prediction) for bet in bets])) == 1
    assert len(bets[0].bet) == len(bets[0].prediction)
    prediction_len: int = len(bets[0].prediction)
    weighted_sum: List[float] = [0. for _ in range(prediction_len)]
    total_weights: List[float] = [0. for _ in range(prediction_len)]
    bet: WeightedBet[A, S]
    for bet in bets:
        assert len(bet.prediction) == prediction_len
        t: int
        prediction_at_t: float
        weight_at_t: float
        for t, (prediction_at_t, weight_at_t) in enumerate(zip(bet.prediction, bet.weight())):
            weighted_sum[t] += prediction_at_t * weight_at_t
            total_weights[t] += weight_at_t

    return [w_sum / total_w for w_sum, total_w in zip(weighted_sum, total_weights)]


def argmax(args: Sequence[T], fn: Callable[[T], float]) -> Optional[T]:
    """
    Breaks ties with the first element found
    Parameters
    ----------
    args
    fn

    Returns
    -------

    """
    maximum: Optional[float] = None
    max_arg: Optional[T] = None
    arg: T
    for arg in args:
        value: float = fn(arg)
        if maximum is None or value > maximum:
            maximum = value
            max_arg = arg
    return max_arg


def dict_argmax(dictionary: Dict[T, float]) -> Optional[T]:
    return argmax(list(dictionary.keys()), lambda key: dictionary[key])

# TODO: maybe make a weighted-specific file??
def map_vals(weighted_elements: Iterable[Weighted], fn: Callable[[float], float]) -> List[Weighted]:
    return [Weighted(w.weight, fn(w.val)) for w in weighted_elements]

def total_weight(
        weighted_vals: Iterable[Weighted]) -> float:
    return float(np.sum([w.weight for w in weighted_vals]))

def normalize_weight(
        weighted_vals: Sequence[Weighted]) -> List[Weighted]:
    # Can't be an iterable because we have to go through it twice
    total: float = total_weight(weighted_vals)
    return [Weighted(w.weight / total, w.val) for w in weighted_vals]

def weighted_mean(
        weighted_vals: List[Weighted]) -> float:
    weighted_vals = normalize_weight(weighted_vals)
    # TODO: could be spend up with np.mean???
    return float(np.sum(weighted_vals))

def weighted_quartile(
        weighted_losses: List[Weighted],
        quartile: float = 0.95) -> float:
    """
    Returns the lowest loss observed such that quartile% of the losses (by weight)
    have lower loss than it

    Parameters
    ----------
    weighted_losses
    quartile: List[Tuple[float, float]]
        [(weight >= 0, loss >= 0)]

    Returns
    -------
    loss: float >= 0
        the 95th quartile loss
    """
    assert 0 < quartile <= 1
    sorted_indices: np.ndarray = np.argsort([w.val for w in weighted_losses])
    weighted_losses = [weighted_losses[i] for i in sorted_indices]
    # sort in order of increasing loss

    weighted_losses = normalize_weight(weighted_losses)
    # sort the losses in order of increasing loss

    weight_so_far: float = 0.  # observed weight we have seen so far
    for w in weighted_losses:
        weight_so_far += w.weight
        if weight_so_far >= quartile:
            return w.val
        # keep going until we've gone through quartile% of the losses by weight
        # then return the first loss we see
    raise Exception("should never get here")