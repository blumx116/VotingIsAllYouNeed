from abc import ABC, abstractmethod
from copy import copy
from typing import List, Callable, Optional, Generic

from VIAYN.project_types import Agent, A, S, ActionBet

class VotingMechanism(Generic[S], ABC):
    @abstractmethod
    def vote(self, state: S) -> float:
        ...

class StaticVotingMechanism(Generic[S], VotingMechanism[S]):
    def __init__(self,
            constant_vote: float):
        self.constant_vote: float = constant_vote

    def vote(self, state: S) -> float:
        return self.constant_vote

class BettingMechanism(Generic[A, S], ABC):
    @abstractmethod
    def bet(self, state: S, action: A, money: float) -> ActionBet:
        ...

class StaticBettingMechanism(Generic[A, S], BettingMechanism[A, S]):
    def __init__(self,
            constant_bet: List[float],
            constant_prediction: List[float]):
        for bet in self.constant_bet:
            assert 0 <= bet <= 1
        self.constant_bet: List[float] = constant_bet
        self.constant_prediction: List[float] = constant_prediction

    def bet(self, state: S, action: A, money: float) -> ActionBet:
        prediction: List[float] = copy(self.constant_prediction)
        bet: List[float] = copy(self.constant_bet)
        return ActionBet(bet=bet, prediction=prediction)
