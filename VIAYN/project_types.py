# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-11-27 20:48:03
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-06 18:37:04

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, List, Iterable, Dict, Tuple, Callable


@dataclass
class Action:
    description: Optional[str]

VoteBoundGetter = Callable[[int], float]

class VoteRange(ABC):

    @staticmethod
    @abstractmethod
    def contains(value: float) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def maxVote() -> float:
        ...
    
    @staticmethod
    @abstractmethod
    def minVote() -> float:
        ...

A = TypeVar("A", bound=Action)  # ActionType
S = TypeVar("S")  # StateType
B = TypeVar("B")  # BetAggregationType


@dataclass(frozen=True)
class ActionBet:
    bet: List[float]
    prediction: List[float]


class Agent(Generic[A, S], ABC):
    @abstractmethod
    def vote(self,
            state: S) -> float:
        ...

    @abstractmethod
    def bet(self,
            state: S,
            action: A,
            money: float) -> ActionBet:
        ...


@dataclass(frozen=True)
class WeightedBet(Generic[A, S]):
    bet: List[float]  # bij
    prediction: List[float]  # pij
    action: A  # j
    money: float  # m_i^(t)
    cast_by: Agent[A, S]

    def weight(self) -> List[float]:
        return [bet * self.money for bet in self.bet]

    def to_action_bet(self) -> ActionBet:
        return ActionBet(bet=self.bet, prediction=self.prediction)


class Environment(Generic[A, S]):
    @abstractmethod
    def step(self, action: A) -> None:
        ...

    @abstractmethod
    def actions(self) -> Iterable[A]:
        ...

    @abstractmethod
    def state(self) -> S:
        ...

    @abstractmethod
    def done(self) -> bool:
        ...

    @abstractmethod
    def seed(self, random_seed: int = None):
        ...

    @abstractmethod
    def reset(self) -> S:
        ...


@dataclass(frozen=True)
class HistoryItem(Generic[A, S]):
    selected_action: A  # jstar
    predictions: Dict[A, List[WeightedBet[A, S]]]
    t_enacted: int  # >= 0

    def available_actions(self) -> List[A]:
        return list(self.predictions.keys())


class Configuration(Generic[A, S], ABC):
    @abstractmethod
    def validate_bet(self,
            bet: WeightedBet[A, S]) -> bool:
        return True


class VotingConfiguration(Generic[A, S], Configuration[A, S], ABC):
    vote_range: VoteRange
    n_agents: int

    @abstractmethod
    def aggregate_votes(self,
            votes: List[float]) -> float:
        ...

    @abstractmethod
    def set_n_agents(self,
            n_agents: int) -> None:
        self.n_agents = n_agents

    @abstractmethod
    def max_possible_vote_total(self, dt: int = 0) -> float:
        """

        :param dt: int
            Timestep offset from current timestep. Useful because
            valid votes may change as number of agents is expected
            to grow
        :return: max: float
            the maximum possible value that could be achieved if
            all agents voted the maximum score
        """
        ...

    @abstractmethod
    def min_possible_vote_total(self, dt: int = 0) -> float:
        """

        :param  dt: int
            Timestep offset from current timestep. Useful because
            valid votes may change as number of agents is expected
            to grow
        :return: min: float
            the minimum possible value that could be achieved if
            all agents voted the minimum score
        """
        ...

    @staticmethod
    def is_valid_prediction(
            prediction: List[float]) -> bool:
        ...


class PolicyConfiguration(Generic[A, B, S], Configuration[A, S], ABC):
    # WeightedBet = WeightedBet[ActionType]
    @abstractmethod
    def aggregate_bets(self,
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, B]:
        ...

    @abstractmethod
    def select_action(self,
            aggregate_bets: Dict[A, B]) -> A:
        ...

    @abstractmethod
    def action_probabilities(self,
            aggregate_bets: Dict[A, B]) -> Dict[A, float]:
        ...


class PayoutConfiguration(Generic[A, S], Configuration[A, S]):
    @abstractmethod
    def calculate_loss(self,
            bet_to_evaluate: ActionBet,
            t_cast_on: int,  # timestep info let us look up in the array
            t_current: int,  # which prediction is for this timestep
            welfare_score: float) -> float:
        ...

    @abstractmethod
    def calculate_payout_from_loss(self,
            loss_to_evaluate: float,
            all_losses: List[Tuple[float, float]], # [(weight, loss)]
            t_cast_on: int, # timestep info lets us discount by timestep
            t_current: int,
            action_bet_on: A,
            action_selected: A) -> float:
        ...

    @abstractmethod
    def calculate_all_payouts(self,
            record: HistoryItem,
            welfare_score: float,
            t_current: int) -> Dict[Agent[A, S], float]:
        ...

    @staticmethod
    def _is_valid_bet_(bet: List[float]) -> bool:
        bet_at_timestep: float
        for bet_at_timestep in bet:
            if not (0 <= bet_at_timestep <= 1):
                return False
        return True


@dataclass(frozen=True)
class SystemConfiguration(Generic[A, B, S], Configuration[A, S]):
    voting_manager: VotingConfiguration[A, S]
    policy_manager: PolicyConfiguration[A, B, S]
    payout_manager: PayoutConfiguration[A, S]

    def validate_bet(self,
            bet: WeightedBet[A, S]) -> bool:
        return self.voting_manager.validate_bet(bet) and \
                self.policy_manager.validate_bet(bet) and \
                self.payout_manager.validate_bet(bet)
