# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-12-06 14:44:56
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-06 14:54:12
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Callable, Optional, Generic

import numpy as np
from numpy.random import Generator, default_rng

from VIAYN.project_types import Agent, A, S, ActionBet


"""
Overall format of this file is as follows.
For all currently existing agents, betting, voting & predicting are completely separate processes.
As a result VotingMechanism, BetSelectionMech & PredictionSelectionMech are abstract classes that these 
can be delegated to. A couple of instantiations of these are provided

Because betting & predicting may potentially influence each other, BettingMechanism is also provided
as an abstract class that can do both at once. CompositeBettingMechanism just combines & delegates to
BetSelectionMech & PredictionSelectionMech.

The general process for creating a CompositeAgent is as follows.
Create a BettingMechanism by combining  BetSelectionMech & PredictionSelectionMech.
Create a CompositeAgent by combining VotingMechanism & BettingMechanism.

Yes, I know the naming is awful.
"""

class VotingMechanism(Generic[S], ABC):
    """
    Abstract class for independent subcomponent of an agent that handles voting
    NOTE: not all agents must use voting mechanism, it's just a useful tool
    """
    @abstractmethod
    def vote(self, state: S) -> float:
        """
        Creates a vote based on the current state

        Parameters
        ----------
        state: S
            the current state at the time of the vote

        Returns
        -------
        vote: float >= 0
            the vote for how much the agent likes the current state
            subject to rules in voting configuration
        """
        ...


class StaticVotingMechanism(Generic[S], VotingMechanism[S]):
    """
    Subcomponent of agent that results in the agent submitting the same
    vote, regardless of state or timestep
    """
    def __init__(self,
            constant_vote: float):
        """

        Parameters
        ----------
        constant_vote: float >= 0
            the vote that the agent will submit at each timestep
            subject to rules in voting configuration
        """
        self.constant_vote: float = constant_vote

    def vote(self, state: S) -> float:
        """
        Creates a vote based on the current state

        Parameters
        ----------
        state: S
            the current state at the time of the vote

        Returns
        -------
        vote: float >= 0
            the vote for how much the agent likes the current state
            subject to rules in voting configuration
        """
        return self.constant_vote


class BetSelectionMechanism(Generic[A, S], ABC):
    """
    Abstract class for independent subcomponent of an agent that handles betting
    NOTE: not all agents must use betting mechanism, it's just a useful tool
    """
    @abstractmethod
    def select_bet_amount(self, state: S, action: A, money: float) -> List[float]:
        """
                Creates a bet based on the current state, the action being bet on
                    and the current amount of money the agent has

                Parameters
                ----------
                state: S
                    the current state at the time of the bet
                action: A
                    the action the agent is betting on
                money: float
                    the amount of money the agent has at the time of the bet

                Returns
                -------
                bet: List[float]
                    how much money to bet on each timestep of the prediction
                    total should sum up to less than 1, the percentage of the agent's
                    current total money
                """
        ...


class StaticBetSelectionMech(Generic[A, S], BetSelectionMechanism [A, S]):
    def __init__(self,
            constant_bet: List[float]):
        """

        Parameters
        ----------
        constant_bet: List[float]
            the percentage of the agent's money to bet on each timestep of every prediction
            each percentage should be represented as a value between 0 and 1
        """
        for bet in constant_bet:
            assert 0 <= bet <= 1
        self.constant_bet: List[float] = constant_bet

    def select_bet_amount(self, state: S, action: A, money: float) -> List[float]:
        """
        Creates a bet based on the current state, the action being bet on
        and the current amount of money the agent has

        Parameters
        ----------
        state: S
            ignored
        action: A
            ignored
        money: float
            ignored

        Returns
        -------
        bet: List[float]
            percentage of money bet @ for each timestep
            constant, as specified in constructor
        """
        return copy(self.constant_bet)
        # copies so that the og bet isn't changed if someone edits the bet


class PredictionSelectionMechanism(Generic[A, S], ABC):
    """
    Abstract class for independent subcomponent of an agent that handles betting
    NOTE: not all agents must use betting mechanism, it's just a useful tool
    """
    @abstractmethod
    def select_prediction(self, state: S, action: A, money: float) -> List[float]:
        ...


class RNGUniforPredSelectionMech(Generic[A, S], PredictionSelectionMechanism[A, S]):
    """
    Uniformly selects a prediction within the valid range for each timestep.
    Intended to be used as part of a CompositeAgent
    """
    def __init__(self,
            tsteps_per_prediction: int,
            min_possible_prediction: Callable[[int], float],
            max_possible_prediction: Callable[[int], float],
            random_seed: int):
        """


        Parameters
        ----------
        tsteps_per_prediction: int > 0
            the number of timesteps that will be in each prediction
            (i.e. the length of the result of select_prediction)
        min_possible_prediction: Callable[[int], float]
            gets the minimum possible prediction for dt: int steps in the future
            called for each element of the prediction. OK to return inf & nan
        max_possible_prediction: Callable[[int], float]
            gets the maximum possible prediction for dt: int steps in the future
            called for each element of the prediction. OK to return inf & nan
        random_seed: int
            seed for random number generator
        """
        self.tsteps_per_prediction: int = tsteps_per_prediction  # length of prediction vector
        self.min_possible_prediction: Callable[[int], float] = min_possible_prediction
        self.max_possible_prediction: Callable[[int], float] = max_possible_prediction
        # any prediction that is higher than the maximum vote total for a given timestep
        # or lower than the minimum is invalid, so we keep within that range
        self.random: Generator = default_rng(random_seed)

    def select_prediction(self, state: S, action: A, money: float) -> List[float]:
        """
        Makes a random prediction starting @ the next timestep
        Selected uniformly from range of possible predictions at that timestep
        Parameters
        ----------
        state: S
            ignored
        action: A
            ignored
        money: float
            ignored
        Returns
        -------
        prediction: List[float]
            the predictions that the agent wants to put in the best at the current timestep
        """
        # TODO : bascially duplicate code with UniformBettingMechanism.bet
        prediction: List[float] = [0. for _ in range(self.tsteps_per_prediction)]
        # TODO : I think dt should start at one, as we'll always predicting about the future
        for dt in range(self.tsteps_per_prediction):
            # note that dt is passed in for each timestep, as max & min may change by time
            low: float = self.min_possible_prediction(dt)
            if not np.isfinite(low):
                # TODO: put these constants somewhere
                low = -100.
            # lowe bound for range is minimum possible : -100 if no minimum
            high: float = self.max_possible_prediction(dt)
            if not np.isfinite(high):
                high = 100.
            # upper bound is max possible: 100 otherwise
            prediction[dt] = self.random.uniform(low=low, high=high)
        return prediction


class StaticPredSelectionMech(Generic[A, S], PredictionSelectionMechanism[A, S]):
    """
    Places the same prediction for every timestep.
    No guarantees that that prediction is always valid
    """
    def __init__(self,
            constant_prediction: List[float]):
        """

        Parameters
        ----------
        constant_prediction: List[float]
            the predictiosn that will be returned each timestep
        """
        self.constant_prediction: List[float] = constant_prediction

    def select_prediction(self, state: S, action: A, money: float) -> List[float]:
        """
        Returns the same constant predictions

        Parameters
        ----------
        state: S
            ignored
        action: A
            ignored
        money: float
            ignored

        Returns
        -------
        prediction: List[float]
            the same constant prediction every time
        """
        return copy(self.constant_prediction)


class BettingMechanism(Generic[A, S], ABC):
    """
    Abstract class for independent subcomponent of an agent that handles betting & predictions
    NOTE: not all agents must use betting mechanism, it's just a useful tool
    """
    @abstractmethod
    def bet(self, state: S, action: A, money: float) -> ActionBet:
        """

        Parameters
        ----------
        state: S
            the state when the bet is being cast
        action: A
            the action that is being bet on
        money: float
            the amount of money when the bet is being case
        Returns
        -------
        bet: ActionBet
            contains both the bet amount & the prediction
            TODO: find better terminology for this
        """
        ...


class CompositeBettingMechanism(Generic[A, S], BettingMechanism[A, S]):
    """
    Composite class that just delegates to bet_selection & prediction_selection
    """
    def __init__(self,
            bet_selection: BetSelectionMechanism[A, S],
            prediction_selection: PredictionSelectionMechanism[A, S]):
        self.bet_selection_mech: BetSelectionMechanism[A, S] = \
            bet_selection
        self.prediction_selection_mech: PredictionSelectionMechanism[A, S] = \
            prediction_selection

    def bet(self, state: S, action: A, money: float) -> ActionBet:
        return ActionBet(
            bet=self.bet_selection_mech.select_bet_amount(state, action, money),
            prediction=self.prediction_selection_mech.select_prediction(state, action, money))


class UniformBettingMechanism(Generic[A, S], BettingMechanism[A, S]):
    """
    TODO: this should use RNGUniformPredSelectionMech
    """
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
        self.random = default_rng(random_seed)

    def bet(self, state: S, action: A, money: float) -> ActionBet:
        """
        Essentially delegates to RNGUniformPredSelectionMech and StaticBetSelectionMech
        Parameters
        ----------
        state: S
            ignored
        action: A
            ignored
        money: float
            ignored

        Returns
        -------
        bet: ActionBet
            bet amount & prediction
        """
        bet: List[float] = copy(self.constant_bet)
        prediction: List[float] = [0. for _ in range(self.tsteps_per_prediction)]
        for dt in range(self.tsteps_per_prediction):
            low: float = self.min_possible_prediction(dt)
            if not np.isfinite(low):
                low = 0
            high: float = self.max_possible_prediction(dt)
            if not np.isfinite(high):
                high = 10.
            prediction[dt] = self.random.uniform(low=low, high=high)
        return ActionBet(bet=bet, prediction=prediction)


class CompositeAgent(Generic[A, S], Agent[A, S]):
    def __init__(self,
            betting_mechanism: BettingMechanism[A, S],
            voting_mechanism: VotingMechanism[S]):
        """
        Just delegates to the respective mechanisms

        Parameters
        ----------
        betting_mechanism: BettingMechanism[A, S]
            used for bet()
        voting_mechanism: VotingMechanism[S]
            used for vote()
        """
        self.betting_mechanism: BettingMechanism = betting_mechanism
        self.voting_mechanism: VotingMechanism = voting_mechanism

    def vote(self, state: S) -> float:
        return self.voting_mechanism.vote(state)

    def bet(self, state: S, action: A, money: float) -> ActionBet:
        return self.betting_mechanism.bet(state, action, money)
