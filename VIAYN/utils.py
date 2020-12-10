from copy import copy
from typing import Dict, List, Sequence, TypeVar, Optional, Callable

from VIAYN.project_types import A, S, WeightedBet

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
