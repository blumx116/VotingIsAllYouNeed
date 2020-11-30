from abc import ABC, abstractmethod
from typing import NamedTuple, Generic, TypeVar, Dict, List, Optional, Union, Iterable

class Action:
    description: Optional[str]

class VoteRange(ABC):
    def contains(self, value: float) -> bool:
        ...

ActionType = TypeVar("ActionType", bound=Action)

VoteType = TypeVar("VoteType") # V

Prediction = Union[float, List[float]]
PredictionType = TypeVar("PredictionType", bound=Prediction)

ActionSpaceType = TypeVar("ActionSpaceType", bound=Iterable[Action])
StateType = TypeVar("StateType")

BetAggregationType = TypeVar("BetAggregationType")


class ActionBet(NamedTuple, Generic[PredictionType]):
    bet: float
    prediction: PredictionType

class Agent(ABC):
    @abstractmethod
    def vote(self, 
            state: StateType) -> float: ...

    @abstractmethod
    def bet(self,
            action: ActionType, 
            state: StateType, 
            money: float) -> ActionBet:
        ...

AgentType = TypeVar("AgentType", bound=Agent)

class WeightedBet(NamedTuple, Generic[PredictionType, ActionType, AgentType]):
    bet: float # bij
    prediction: PredictionType # pij of type K
    action: ActionType # j
    money: float 
    cast_by: AgentType
    t: int = 0
    
    def weight(self) -> float:
        return self.bet * self.money

    
class Environment(ABC, Generic[ActionType, StateType]):
    @abstractmethod
    def step(self, action: ActionType) -> None:
        ...

    @abstractmethod
    def actions(self) -> ActionSpaceType:
        ... # action space???

    @abstractmethod
    def state(self) -> StateType:
        ...

    @abstractmethod
    def done(self) -> bool:
        ...


class HistoryItem(NamedTuple, Generic[ActionType, PredictionType, AgentType]):
    selected_action: ActionType # jstar
    available_actions: List[ActionType] # as
    predictions: Dict[ActionType, WeightedBet]
    t_enacted: int # >= 0

class LossLookup(Generic[ActionType, AgentType]):
    def __init__(self, lookup: Dict[AgentType, Dict[ActionType, float]]):
        self._lookup: Dict[AgentType, Dict[ActionType, float]] = lookup

    def loss_for(self, agent: AgentType, action: ActionType) -> float:
        return self._lookup[agent][action]

class VotingConfiguration(ABC, Generic[VoteType]):
    @abstractmethod
    def aggregate_votes(self, votes: List[VoteType]) -> float:
        ...

class PolicyConfiguration(ABC, Generic[PredictionType, ActionType, AgentType, BetAggregationType]):
    WeightedBet = WeightedBet[PredictionType, ActionType, AgentType]
    # A
    @abstractmethod
    def aggregate_bets(self, predictions: Dict[ActionType, WeightedBet], actual: float) -> Dict[ActionType, BetAggregationType]:
        ...

    # pi
    @abstractmethod
    def select_action(self, aggregated_bets: Dict[ActionType, BetAggregationType]) -> ActionType:
        ...

    # strictly pi as described in math
    @abstractmethod
    def action_probabilities(self, aggregate_bets: Dict[ActionType, BetAggregationType]) -> Dict[ActionType, float]:
        ... # returns probability distribution over actions

class PayoutConfiguration(ABC, Generic[PredictionType, ActionType, AgentType]):
    WeightedBet = WeightedBet[PredictionType, ActionType, AgentType]
    # f
    @abstractmethod
    def received_money(self, bets: Dict[AgentType, WeightedBet], losses: LossLookup) -> Dict[AgentType, float]:
        ...

    # L
    @abstractmethod
    def calculate_loss(self, predictions: Dict[ActionType, WeightedBet], actual: float) -> LossLookup:
        ...


class SystemConfiguration(NamedTuple, Generic[VoteType, PredictionType, ActionType, AgentType, BetAggregationType]):
    voting_manager: VotingConfiguration[VoteType]
    policy_manager: PolicyConfiguration[PredictionType, ActionType, AgentType, BetAggregationType]
    payout_manager: PayoutConfiguration[PredictionType, ActionType, AgentType]