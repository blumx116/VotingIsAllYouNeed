# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-11-27 20:48:03
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-03 20:48:51

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, List, Iterable, Dict


@dataclass
class Action:
    description: Optional[str]


class VoteRange(ABC):
    @abstractmethod
    def contains(self, value: float) -> bool:
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
    available_actions: List[A]  # as
    predictions: Dict[A, WeightedBet[A, S]]
    t_enacted: int  # >= 0


@dataclass(frozen=True)
class LossLookup(Generic[A]):
    # Agent = Agent[ActionType, StateType]
    lookup: Dict[Agent, Dict[A, float]]

    def loss_for(self, agent: Agent, action: A) -> float:
        return self.lookup[agent][action]  # is this worth having???


class VotingConfiguration(ABC):
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
            prediction: List[float])-> bool:
        ...


class PolicyConfiguration(Generic[A, B], ABC):
    # WeightedBet = WeightedBet[ActionType]
    @abstractmethod
    def aggregate_bets(self,
            predictions: Dict[A, WeightedBet[A, S]],
            actual: float) -> Dict[A, B]:
        ...

    @abstractmethod
    def select_action(self,
            aggregate_bets: Dict[A, B]) -> A:
        ...

    @abstractmethod
    def action_probabilities(self,
            aggregate_bets: Dict[A, B]) -> Dict[A, float]:
        ...


class PayoutConfiguration(Generic[A]):
    @abstractmethod
    def calculate_payouts_from_losses(self,
            bets:  Dict[Agent, WeightedBet[A, S]],
            losses: LossLookup[A]) \
            -> Dict[Agent, float]:
        ...

    # L
    @abstractmethod
    def calculate_loss(self,
            predictions: Dict[A, Dict[Agent, WeightedBet[A, S]]],
            actual: float) -> LossLookup[A]:
        ...

    @abstractmethod
    def calculate_payouts(self,
            predictions: Dict[A, Dict[Agent, WeightedBet[A, S]]],
            actual:  float) -> Dict[Agent, float]:
        ...

    @staticmethod
    def _is_valid_bet_(bet: List[float]) -> bool:
        bet_at_timestep: float
        for bet_at_timestep in bet:
            if not (0 <= bet_at_timestep <= 1):
                return False
        return True


@dataclass(frozen=True)
class SystemConfiguration(Generic[A, B]):
    voting_manager: VotingConfiguration
    policy_manager: PolicyConfiguration[A, B]
    payout_manager: PayoutConfiguration[A]
