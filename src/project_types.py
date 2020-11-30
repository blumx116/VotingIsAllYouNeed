from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, Union, List, Iterable, Dict

@dataclass
class Action:
    description: Optional[str]

class VoteRange(ABC):
    def contains(self, value: float) -> bool:
        ...

ActionType = TypeVar("ActionType", bound=Action)

Prediction = Union[float, List[float]]
PredictionType =TypeVar("PredictionType", bound=Prediction)

ActionSpaceType = Iterable[ActionType]
StateType = TypeVar("StateType")

BetAggregationType = TypeVar("BetAggregationType")

@dataclass(frozen=True)
class ActionBet(Generic[PredictionType]):
    bet: float 
    prediction: PredictionType

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
class WeightedBet(Generic[ActionType, PredictionType]):
    bet: float # bij
    prediction: PredictionType # pij
    action: ActionType # j
    money: float # m_i^(t)
    cast_by: Agent[ActionType, PredictionType]

    def weight(self) -> float:
        return self.bet * self.money

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
class HistoryItem(Generic[ActionType, PredictionType]):
    selected_action: ActionType # jstar
    available_actions: List[ActionType] # as
    predictions: Dict[ActionType, WeightedBet[ActionType, PredictionType]]
    t_enacted: int # >= 0

@dataclass(frozen=True)
class LossLookup(Generic[ActionType, PredictionType]):
    # Agent = Agent[ActionType, PredictionType]
    lookup: Dict[Agent, Dict[ActionType, float]]

    def loss_for(self, agent: Agent, action: ActionType) -> float:
        return self.lookup[agent][action] # is this worth having???

class VotingConfiguration(ABC):
    vote_range: VoteRange
    @abstractmethod
    def aggregate_votes(self,
            vots: List[float]) -> float:
        ...

class PolicyConfiguration(Generic[ActionType, BetAggregationType, PredictionType], ABC):
    # WeightedBet = WeightedBet[ActionType, PredictionType]
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

class PayoutConfiguration(Generic[ActionType, PredictionType]):
    # WeightedBet = WeightedBet[ActionType, PredictionType]
    # Agent = Agent[ActionType, PredictionType]
    # f 
    @abstractmethod
    def calculate_payouts(self, 
            bets: Dict[Agent, WeightedBet], 
            losses: LossLookup[ActionType, PredictionType]) \
            -> Dict[Agent, float]:
        ...

    @abstractmethod
    def calculate_loss(self,
            predictions: Dict[ActionType, WeightedBet],
            actual: float) -> LossLookup[ActionType, PredictionType]:
        ...

@dataclass(frozen=True)
class SystemConfiguration(Generic[ActionType, BetAggregationType, PredictionType]):
    voting_manager: VotingConfiguration
    policy_manager: PolicyConfiguration[ActionType, BetAggregationType, PredictionType]
    payout_manager: PayoutConfiguration[ActionType, PredictionType]