from abc import ABC, abstractmethod
from copy import copy
from typing import List, Callable, Optional, Generic

import numpy as np

from VIAYN.project_types import Agent, StateType, ActionType, ActionBet


class VotingMechanism(Generic[StateType], ABC):
    @abstractmethod
    def vote(self, state: StateType) -> float:
        ...


class StaticVotingMechanism(Generic[StateType], VotingMechanism[StateType]):
    def __init__(self,
            constant_vote: float):
        self.constant_vote: float = constant_vote

    def vote(self, state: StateType) -> float:
        return self.constant_vote


class BetSelectionMechanism(Generic[StateType, ActionType], ABC):
    @abstractmethod
    def select_bet_amount(self, state: StateType, action: ActionType, money: float) -> List[float]:
        ...


class StaticBetSelectionMech(Generic[StateType, ActionType], BetSelectionMechanism):
    def __init__(self,
            constant_bet: List[float]):
        for bet in constant_bet:
            assert 0 <= bet <= 1
        self.constant_bet: List[float] = constant_bet

    def select_bet_amount(self, state: StateType, action: ActionType, money: float) -> List[float]:
        return copy(self.constant_bet)


class PredictionSelectionMechanism(Generic[StateType, ActionType], ABC):
    @abstractmethod
    def select_prediction(self, state: StateType, action: ActionType, money: float) -> List[float]:
        ...


class RNGUniforPredSelectionMech(Generic[StateType, ActionType], PredictionSelectionMechanism[StateType, ActionType]):
    def __init__(self,
            tsteps_per_prediction: int,
            min_possible_prediction: Callable[[int], float],
            max_possible_prediction: Callable[[int], float],
            random_seed: int):
        self.tsteps_per_prediction: int = tsteps_per_prediction
        self.min_possible_prediction: Callable[[int], float] = min_possible_prediction
        self.max_possible_prediction: Callable[[int], float] = max_possible_prediction
        self.random = np.random.RandomState(random_seed)

    def select_prediction(self, state: StateType, action: ActionType, money: float) -> List[float]:
        return [self.random.uniform(low=self.min_possible_prediction(dt), high=self.max_possible_prediction(dt))
         for dt in range(self.tsteps_per_prediction)]


class StaticPredSelectionMech(Generic[StateType, ActionType], PredictionSelectionMechanism[StateType, ActionType]):
    def __init__(self,
            constant_prediction: List[float]):
        self.constant_prediction: List[float] = constant_prediction

    def select_prediction(self, state: StateType, action: ActionType, money: float) -> List[float]:
        return copy(self.constant_prediction)

class BettingMechanism(Generic[StateType, ActionType], ABC):
    @abstractmethod
    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        ...


class CompositeBettingMechanism(BettingMechanism):
    def __init__(self,
            bet_selection: BetSelectionMechanism[StateType, ActionType],
            prediction_selection: PredictionSelectionMechanism[StateType, ActionType]):
        self.bet_selection_mech: BetSelectionMechanism[StateType, ActionType] = \
            bet_selection
        self.prediction_selection_mech: PredictionSelectionMechanism[StateType, ActionType] = \
            prediction_selection

    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        return ActionBet(
            bet=self.bet_selection_mech.select_bet_amount(state, action, money),
            prediction=self.prediction_selection_mech.select_prediction(state, action, money))


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
            min_possible_prediction: Callable[[int], float],
            max_possible_prediction: Callable[[int], float],
            random_seed: int):
        self.tsteps_per_prediction: int = tsteps_per_prediction
        self.constant_bet: List[float] = constant_bet
        self.min_possible_prediction: Callable[[int], float] = min_possible_prediction
        self.max_possible_prediction: Callable[[int], float] = max_possible_prediction
        self.random = np.random.RandomState(random_seed)

    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        bet: List[float] = copy(self.constant_bet)
        prediction: List[float] = \
            [self.random.uniform(low=self.min_possible_prediction(dt), high=self.max_possible_prediction(dt))
                for dt in range(self.tsteps_per_prediction)]
        return ActionBet(bet=bet, prediction=prediction)


class CompositeAgent(Agent):
    def __init__(self,
            betting_mechanism: BettingMechanism,
            voting_mechanism: VotingMechanism):
        self.betting_mechanism: BettingMechanism = betting_mechanism
        self.voting_mechanism: VotingMechanism = voting_mechanism

    def vote(self, state: StateType) -> float:
        return self.voting_mechanism.vote(state)

    def bet(self, state: StateType, action: ActionType, money: float) -> ActionBet:
        return self.betting_mechanism.bet(state, action, money)
