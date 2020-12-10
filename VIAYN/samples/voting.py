from math import sqrt
from typing import Generic, List

import numpy as np


from VIAYN.project_types import VoteRange, VotingConfiguration, A, S, WeightedBet
from VIAYN.samples.vote_ranges import FiveStarVoteRange, BinaryVoteRange, ZeroToTenVoteRange


class ClassicalVotingConfig(Generic[A, S], VotingConfiguration[A, S]):
    def __init__(self):
        self.n_agents: int = 0

    vote_range: VoteRange = BinaryVoteRange() # technically instantiating a static class
    # but makes types work out without being nasty

    def validate_bet(self, bet: WeightedBet[A, S]) -> bool:
        return self.is_valid_prediction(bet.prediction)

    def aggregate_votes(self,
            votes: List[float]) -> float:
        return sum(votes)

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
        return self.n_agents

    def min_possible_vote_total(self, dt: int = 1) -> float:
        """

        :param  dt: int
            Timestep offset from current timestep. Useful because
            valid votes may change as number of agents is expected
            to grow
        :return: min: float
            the minimum possible value that could be achieved if
            all agents voted the minimum score
        """
        return 0

    def is_valid_prediction(self,
            prediction: List[float]) -> bool:
        min_pred: float = self.min_possible_vote_total()
        max_pred: float = self.max_possible_vote_total()
        prediction_at_t: float
        for prediction_at_t in prediction:
            if not min_pred <= prediction_at_t <= max_pred:
                return False
        return True


class RecommendedVoting(Generic[A, S], VotingConfiguration[A, S]):
    def __init__(self):
        self.n_agents: int = 0

    vote_range: VoteRange = ZeroToTenVoteRange()

    def validate_bet(self, bet: WeightedBet[A, S]) -> bool:
        return self.is_valid_prediction(bet.prediction)

    def aggregate_votes(self,
            votes: List[float]) -> float:
        return sum(map(sqrt, votes))

    def max_possible_vote_total(self, dt: int = 0) -> float:
        # TODO: implement way to account for expected growth in n_agents
        return sqrt(self.vote_range.maxVote()) * self.n_agents

    def min_possible_vote_total(self, dt: int = 1) -> float:
        return 0.

    def is_valid_prediction(self,
            prediction: List[float]) -> bool:
        dt: int
        prediction_at_t: float
        for dt, prediction_at_t in enumerate(prediction):
            if not (self.min_possible_vote_total(dt)
                    <= prediction_at_t
                    <= self.max_possible_vote_total(dt)):
                return False
        return True

