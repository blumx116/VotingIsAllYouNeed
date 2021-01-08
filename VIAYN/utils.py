# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-12-10 14:54:22
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:59:06
from copy import copy
from typing import Dict, List, Sequence, TypeVar, Optional, Callable, Any
from decimal import Decimal

import numpy as np
from numpy.random import Generator

from VIAYN.project_types import A, S, WeightedBet

T = TypeVar("T")
U = TypeVar("U", int, float, complex, str)


def add_dictionaries(
        first: Dict[T, U],
        second: Dict[T, U]) -> Dict[T, U]:
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
    result: Dict[T, U] = copy(first)
    key: T
    value: U
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


def dict_argmax(dictionary: Dict[T, float], rng: Optional[Generator] = None) -> Optional[T]:
    if rng is None:
        rng = np.random.default_rng()
    inputs: List[T] = list(dictionary.keys())
    rng.shuffle(inputs)
    return argmax(inputs, lambda key: dictionary[key])

def iterable_matches(item: Sequence, filter: Sequence) -> int:
    """
    This is a bit of a weird function to match sequences with filters.
    It's currently intended to be used for look-up in a dictionary with optional arguments.

    Here are some examples:
    iterable_matches((s1, a1), (s1, a1)) => 2
    iterable_matches((s1, a1), (s1, a2)) => -1
    iterable_matches((s1, a1), (s1, None)) => 1

    in short, it checks if every non-None element of the filter is equal to the
    corresponding element of the item. If not, it returns -1. If yes, then it
    returns the number of non-None values in the filter.

    This is useful for making policies of varying strictness for look-up in a dictionary.
    Given a lookup of the form Dict[filter: Tuple, value: whatever], you can iterate through the items
    to find the key that matches the strictes filter (the filter that returns the highest value from
    iterable matches). See policy_lookup

    Parameters
    ----------
    item: Sequence
        item to check
    filter: Sequence
        each element of the filter should have the same type as
        the corresponding element of item or be None

    Returns
    -------
    n_matches: int
        returns -1 if there are any non-None elements of filter
        that 'item' doesn't match.
        Otherwise, returns the number of non-None elements of filter
    """
    assert len(item) == len(filter)
    assert len(filter) > 0
    n_matches: int = 0
    for item_elem, filter_elem in zip(item, filter):
        if filter_elem is not None:
            if item_elem == filter_elem:
                n_matches += 1
            else:
                return -1
    return n_matches


K = TypeVar("K", bound=Sequence)
V = TypeVar("V")


def behaviour_lookup_from_dict(key: K, lookup: Dict[K, V]) -> Optional[V]:
    """
    Assumes the key argument and the keys of lookup are both sequences of the same types.
    Keys of lookup are interpreted as filters

    Iterates through the elements of lookup and tries to find the most restrictive filter that
    still matches the key argument.

    Returns the corresponding value in lookup. If no such value found, returns None

    Parameters
    ----------
    key: K (Sequence)
    lookup: Dict[K, V]]
        keys like key except may contain Nones
        interpreted as filters of varying strictness

    Returns
    -------
    best_match: Optional[V]
         the value corresponding to the strictest filter that
         'key' was able to fulfil. None if no filters were fulfilled
    """
    best_matching_val: Optional[V] = None
    best_match_score: Optional[float] = None

    k: K
    v: V
    for k, v in lookup.items():
        score: float = iterable_matches(key, k)
        if score >= 0:
            if best_match_score is None or \
                    score > best_match_score:
                best_match_score = score
                best_matching_val = v
    return best_matching_val


def is_numeric(val: Any) -> bool:
    return isinstance(val, (float, int, Decimal))
