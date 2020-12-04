from abc import ABC, abstractmethod
from copy import copy
from typing import List, Callable, Optional

import numpy as np

from VIAYN.project_types import Agent, StateType, ActionType, ActionBet

class VotingMechanism(ABC):
    @abstractmethod
    def vote(self, state: StateType) -> float:
        ...

class StaticVotingMechanism(VotingMechanism):
    def __init__(self,
            constant_vote: float):
        self.constant_vote: float = constant_vote

    def vote(self, state: StateType) -> float:
        return self.constant_vote

class BettingMechanism(ABC):
    @abstractmethod
    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        ...

class StaticBettingMechanism(BettingMechanism):
    def __init__(self,
            constant_bet: List[float],
            constant_prediction: List[float]):
        for bet in self.constant_bet:
            assert 0 <= bet <= 1
        self.constant_bet: List[float] = constant_bet
        self.constant_prediction: List[float] = constant_prediction

    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        prediction: List[float] = copy(self.constant_prediction)
        bet: List[float] = copy(self.constant_bet)
        return ActionBet(bet=bet, prediction=prediction)

class UniformBettingMechanism(BettingMechanism):
    def __init__(self,
            constant_bet: List[float],
            tsteps_per_prediction: int,
            min_possible_prediction: Callable[[], float],
            max_possible_prediction: Callable[[], float],
            random_seed: int = None):
        self.tsteps_per_prediction: int = tsteps_per_prediction
        self.constant_bet: List[float] = constant_bet
        self.min_possible_prediction: Optional[Callable[[int], float]] = min_possible_prediction
        self.max_possible_prediction: Optional[Callable[[int], float]] = max_possible_prediction
        self.random = np.random.RandomState(random_seed)


    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        bet: List[float] = copy(self.constant_bet)
        prediction: List[float] = \
            [self.random.uniform(low=self.min_possible_prediction(dt), high=self.max_possible_prediction(dt))
                for dt in range(self.tsteps_per_prediction)]
        return ActionBet(bet=bet, prediction=prediction)

