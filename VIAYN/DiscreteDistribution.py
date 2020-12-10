from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class DiscreteDistribution:
    values: List[float]
    probabilities: List[float]
