# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-11-27 20:48:03
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-11 19:03:36

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, List, Iterable, Dict, Tuple, Callable


@dataclass
class Action:
    """
    base class for all actions
    TODO: possibly useless if description is optional?
    """
    description: Optional[str]


VoteBoundGetter = Callable[[int], float]
# type alias for the type of VotingConfiguration.max_possible_vote_total


class VoteRange(ABC):
    """
    Type of possible voting. e.g. Yes/No = {0, 1},
    5 Stars = {1, 2, 3, 4, 5}, etc.

    Checks whether or not votes comply with this format
    because they are all technically floats
    """
    @staticmethod
    @abstractmethod
    def contains(value: float) -> bool:
        """
        Checks whether or not 'value' is valid for this vote range.
        Parameters
        ----------
        value: float
            the value to be checked

        Returns
        -------
        valid: bool
            whether or not the vote is valid for this VoteRange
        """
        ...


    @staticmethod
    @abstractmethod
    def maxVote() -> float:
        """

        Returns
        -------
        maximum: float
            the maximum possible float that can be cast in this system.
            e.g. 5 for 5star voting
            may be inf or nan
        """
        ...
    
    @staticmethod
    @abstractmethod
    def minVote() -> float:
        """

        Returns
        -------
        maximum: float
            the minimum possible float that can be cast in this system
            e.g. 1 for 5star voting
            may be inf or nan
        """
        ...


A = TypeVar("A", bound=Action)  # ActionType
S = TypeVar("S")  # StateType
B = TypeVar("B")  # BetAggregationType


@dataclass(frozen=True)
class ActionBet:
    """
    Minimal version of a bet containing only the parts that are
    directly under the control of the agent
    """
    bet: List[float]  # percentage of money to be bet @ each timestep (bij)
    prediction: List[float]  # the predictions about the welfare score @ each timestep (pij)

    def __post_init__(self):
        assert len(self.bet) == len(self.prediction)
        assert len(self.bet) != 0
        assert sum(self.bet) <= 1
        for b in self.bet:
            assert b >= 0

            
class AnonymizedHistoryItem:
     pass 


class Agent(Generic[A, S], ABC):
    """
    Minimal interface for an agent to be used with train.py
    Agent's need to be able to do 2 things : predict & bet
    """
    t : int = 0 # initial timestep

    @abstractmethod
    def vote(self,
            state: S) -> float:
        ...
    """
    The agent's vote corresponding to how happy there are with a 
    given state. Subject to the constraints of VoteRange
    """

    @abstractmethod
    def bet(self,
            state: S,
            action: A,
            money: float) -> ActionBet:
        ...
    """
    The agent's bet on what they think the (global) welfare scores will be like
    if action A is taken, given that they are currently @ (global) state and have the
    input amount of money (personally)
    """

    def view(self, info: AnonymizedHistoryItem) -> None:
         self.t += 1


@dataclass(frozen=True)
class WeightedBet(Generic[A, S],ActionBet):
    """
    Like ActionBet, but with additional metadata about the bet
    that the Agent should NOT be able to control
    """
    action: A  # j # the action that the bet was placed on
    money: float  # m_i^(t) # the amount of money the agent has at time of bet
    cast_by: Agent[A, S] # which agent it was cast by

    def __post_init__(self):
        ActionBet.__post_init__(self)
        # check for weighted bet requirements below

    def weight(self) -> List[float]:
        return [bet * self.money for bet in self.bet]
        
class Environment(Generic[A, S]):
    """
    Generic interface that aligns with OpenAI's Gym
    """
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
    selected_action: A  # jstar # the action that was selected @ this timestep
    predictions: Dict[A, List[WeightedBet[A, S]]]
     #all of the predictions made
    t_enacted: int  # >= 0
    # the timestep this was generated (same as when all bets were cast)

    def available_actions(self) -> List[A]:
        return list(self.predictions.keys())


class Configuration(Generic[A, S], ABC):
    """
    Base class for configurations, as all configurations may want to validate
    bets in their own way.
    """
    @abstractmethod
    def validate_bet(self,
            bet: WeightedBet[A, S]) -> bool:
        return True


class VotingConfiguration(Generic[A, S], Configuration[A, S], ABC):
    """
    Handles everything voting related.
         - Checking valid votes (via vote_range)
         - Calculating aggregate votes (via aggregate_votes)
         - checking minimum/maximum possible predictions (via calculating min & max
                possible vote totals, hypothetically)
    """
    vote_range: VoteRange
    n_agents: int

    @abstractmethod
    def aggregate_votes(self,
            votes: List[float]) -> float:
        ...

    @abstractmethod
    def set_n_agents(self,
            n_agents: int) -> None:
        """
        Set the number of agents in the system. Useful for calculating
        what the max/min possible vote totals are

        Parameters
        ----------
        n_agents: int > 0
            the number of agents in the system
        """
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

    @abstractmethod
    def is_valid_prediction(self,
            prediction: List[float]) -> bool:
        ...


class PolicyConfiguration(Generic[A, B, S], Configuration[A, S], ABC):
    # WeightedBet = WeightedBet[ActionType]
    @abstractmethod
    def aggregate_bets(self,
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, B]:
        """
        Computes some aggregate statistics about the predictions made.
        The decisions about what action to take must be made SOLELY off of these
        aggregate statistics

        Parameters
        ----------
        predictions: Dict[A, List[WeightedBet[A, S]]]
            predictions to aggregate statistics for for each action

        Returns
        -------
        stats: Dict[A, B]
            mapping from actions to aggregated stats of the predictions about them
        """
        ...

    @abstractmethod
    def select_action(self,
            aggregate_bets: Dict[A, B]) -> A:
        """
        Based on the statistics about the actions, selects an action to take
        May be non-deterministic

        Parameters
        ----------
        aggregate_bets: Dict[A, B]
            statistics about each action. E.g. weighted mean of predictions => B = float

        Returns
        -------
        action: A
            the selected action
        """
        ...

    @abstractmethod
    def action_probabilities(self,
            aggregate_bets: Dict[A, B]) -> Dict[A, float]:
        """
        Computes/estimates the probability that each action is selected

        Parameters
        ----------
        aggregate_bet: Dict[A, B]
            statistics about each action. E.g. weighted mean of predictions => B = float
s

        Returns
        -------
        probabilities: Dict[A, float]
            the probability that each action will be selected, from 0 to 1
            only one action can be taken
        """
        ...


class PayoutConfiguration(Generic[A, S], Configuration[A, S]):
    """
    Handles how much money each agent should get for their predictions
    """
    @abstractmethod
    def calculate_loss(self,
            bet_to_evaluate: WeightedBet,
            t_cast_on: int,  # timestep info let us look up in the array
            t_current: int,  # which prediction is for this timestep
            welfare_score: float) -> float:
        """
        Calculates 'how far off' the agents were from correctly predicting the welfare_score.
        Mostly useful for example & potential use in calculate_all_payouts

        Parameters
        ----------
        bet_to_evaluate: WeightedBet[A, S]
            the bet that we are calculating the loss on
        t_cast_on: int
            the timestep that the bet was cast on
        t_current: int
            the timestep that the evaluation is being made on
            should be the same as the timestep corresponding to welfare score
        welfare_score

        Returns
        -------
        loss: float >= 0
            how far off the predictions were
        """
        ...

    @abstractmethod
    def calculate_payout_from_loss(self,
            bet_amount_to_evaluate: float,
            loss_to_evaluate: float,
            all_losses: List[Tuple[float, float]], # [(weight, loss)]
            t_cast_on: int, # timestep info lets us discount by timestep
            t_current: int,
            action_bet_on: A,
            action_selected: A) -> float:
        ...

    @abstractmethod
    def calculate_all_payouts(self,
            record: HistoryItem[A, S],
            welfare_score: float,
            t_current: int) -> Dict[Agent[A, S], float]:
        """
        Calculates how much money each agent should get for their bets that were
        cast in record

        Parameters
        ----------
        record: HistoryItem[A, S]
            record of all bets that were cast for a single timestep that are
            now eligible for payouts
        welfare_score: float
            the current welfare score that the agents were trying to predict
        t_current: int
            the timestep @ which the welfare score was recorded

        Returns
        -------
        payouts: Dict[Agent[A, S], float]
            how much money each agent earned from their bets in this record
        """
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
    """
    Small class to batch all of the different configurations together.
    """
    voting_manager: VotingConfiguration[A, S]
    policy_manager: PolicyConfiguration[A, B, S]
    payout_manager: PayoutConfiguration[A, S]

    def validate_bet(self,
            bet: WeightedBet[A, S]) -> bool:
        return self.voting_manager.validate_bet(bet) and \
                self.policy_manager.validate_bet(bet) and \
                self.payout_manager.validate_bet(bet)
