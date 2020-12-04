from abc import ABC, abstractmethod
from typing import List, Callable, Optional

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

    @staticmethod
    def _is_valid_bet_(self, bet: List[float]) -> bool:
        for bet in self.constant_bet:
            if not (0 <= bet <= 1):
                return False
        return True

    @staticmethod
    def _is_valid_prediction(self,
            prediction: List[float],
            minimum):

class StaticBettingMechanism(BettingMechanism):
    def __init__(self,
            constant_bet: List[float],
            constant_prediction: List[float],
            min_possible_prediction: Callable[[], float] = None,
            max_possible_prediction: Callable[[], float] = None):
        for bet in self.constant_bet:
            assert 0 <= bet <= 1
        self.constant_bet: List[float] = constant_bet
        self.constant_prediction: List[float] = constant_prediction
        self.min_possible_pred: Optional[Callable[[], float]] = min_possible_prediction
        self.max_possible_pred: Optional[Callable[[], float]] = max_possible_prediction

    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        min_possible: Optional[float] = self.min_possible_pred() if self.min_possible_pred is not None else None
        max_possible: Oo


class StaticAgent(Agent):
    def __init__(self,
            bet_amount: List[float],
            prediction: List[float],
            vote_amount: float):

        assert 0 <= bet_amount <= 1

        self.bet_amount: List[float] = bet_amount
        self.vote_amount: float = vote_amount
        self.prediction: List[float] = prediction

    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        return ActionBet(bet=self.bet_amount, prediction=self.prediction)

    def vote(self, state: StateType) -> float:
        return self.vote_amount

class DumbAgentWithStaticVote