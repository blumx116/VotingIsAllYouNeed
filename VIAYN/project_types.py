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


ActionType = TypeVar("ActionType", bound=Action)

ActionSpaceType = Iterable[ActionType]
StateType = TypeVar("StateType")

BetAggregationType = TypeVar("BetAggregationType")


@dataclass(frozen=True)
class ActionBet:
    bet: List[float]
    prediction: List[float]


class Agent(Generic[ActionType, StateType], ABC):
    @abstractmethod
    def vote(self,
            state: StateType) -> float:
        ...

    @abstractmethod
    def bet(self,
            state: StateType,
            action: ActionType,
            money: float) -> ActionBet:
        ...


@dataclass(frozen=True)
class WeightedBet(Generic[ActionType]):
    bet: List[float]  # bij
    prediction: List[float]  # pij
    action: ActionType  # j
    money: float  # m_i^(t)
    cast_by: Agent[ActionType, StateType]

    def weight(self) -> List[float]:
        return [bet * self.money for bet in self.bet]


class Environment(Generic[ActionType, StateType]):
    @abstractmethod
    def step(self, action: ActionType) -> None:
        ...

    @abstractmethod
    def actions(self) -> Iterable[ActionType]:
        ...

    @abstractmethod
    def state(self) -> StateType:
        ...

    @abstractmethod
    def done(self) -> bool:
        ...


@dataclass(frozen=True)
class HistoryItem(Generic[ActionType]):
    selected_action: ActionType  # jstar
    available_actions: List[ActionType]  # as
    predictions: Dict[ActionType, WeightedBet[ActionType]]
    t_enacted: int  # >= 0


@dataclass(frozen=True)
class LossLookup(Generic[ActionType]):
    # Agent = Agent[ActionType, StateType]
    lookup: Dict[Agent, Dict[ActionType, float]]

    def loss_for(self, agent: Agent, action: ActionType) -> float:
        return self.lookup[agent][action]  # is this worth having???


class VotingConfiguration(ABC):
    vote_range: VoteRange

    @abstractmethod
    def aggregate_votes(self,
            votes: List[float]) -> float:
        ...

    @abstractmethod
    def max_possible_vote_total(self,
            n_agents: int) -> float:
        ...

    @abstractmethod
    def min_possible_vote_total(self,
            n_agents: int) -> float:
        ...


class PolicyConfiguration(Generic[ActionType, BetAggregationType], ABC):
    # WeightedBet = WeightedBet[ActionType]
    @abstractmethod
    def aggregate_bets(self,
            predictions: Dict[ActionType, WeightedBet],
            actual: float) -> Dict[ActionType, BetAggregationType]:
        ...

    @abstractmethod
    def select_action(self,
            aggregate_bets: Dict[ActionType, BetAggregationType]) -> ActionType:
        ...

    @abstractmethod
    def action_probabilities(self,
            aggregate_bets: Dict[ActionType, BetAggregationType]) -> Dict[ActionType, float]:
        ...


class PayoutConfiguration(Generic[ActionType]):
    # WeightedBet = WeightedBet[ActionType]
    # Agent = Agent[ActionType, StateType]
    # f 
    @abstractmethod
    def calculate_payouts_from_losses(self,
            bets:  Dict[Agent, WeightedBet],
            losses: LossLookup[ActionType]) \
            -> Dict[Agent, float]:
        ...

    # L
    @abstractmethod
    def calculate_loss(self,
            predictions: Dict[ActionType, Dict[Agent, WeightedBet]],
            actual: float) -> LossLookup[ActionType]:
        ...

    @abstractmethod
    def calculate_payouts(self,
            predictions: Dict[ActionType, Dict[Agent, WeightedBet]],
            actual:  float) -> Dict[Agent, float]:
        ...


@dataclass(frozen=True)
class SystemConfiguration(Generic[ActionType, BetAggregationType]):
    voting_manager: VotingConfiguration
    policy_manager: PolicyConfiguration[ActionType, BetAggregationType]
    payout_manager: PayoutConfiguration[ActionType]
