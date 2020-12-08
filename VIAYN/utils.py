from typing import TypeVar, Dict, C
from copy import copy

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
