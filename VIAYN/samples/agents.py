from abc import ABC, abstractmethod
from copy import copy
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
    def _is_valid_bet_(bet: List[float]) -> bool:
        bet_at_timestep: float
        for bet_at_timestep in bet:
            if not (0 <= bet_at_timestep <= 1):
                return False
        return True

    @staticmethod
    def _is_valid_prediction_(
            prediction: List[float],
            minimum: Optional[Callable[[int], float]] = None,
            maximum: Optional[Callable[[int], float]] = None) -> bool:
        for dt, prediction in enumerate(prediction):
            if minimum is not None and prediction < minimum(dt):
                return False
            if maximum is not None and prediction > maximum(dt):
                return False
        return True

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
        prediction: List[float] = copy(self.constant_prediction)
        bet: List[float] = copy(self.constant_bet)
        assert self._is_valid_bet_(bet)
        assert self._is_valid_prediction(prediction,
                minimum=self.min_possible_pred,
                maximum=self.max_possible_pred)
        return ActionBet(bet=bet, prediction=prediction)

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