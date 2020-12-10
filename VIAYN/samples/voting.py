from math import sqrt
from typing import Generic, List

import numpy as np


from VIAYN.project_types import VoteRange, VotingConfiguration, A, S, WeightedBet
from VIAYN.samples.vote_ranges import UnboundedVoteRange, BinaryVoteRange, ZeroToTenVoteRange


class VotingConfigurationBase(Generic[A, S], VotingConfiguration[A, S]):
    def __init__(self,
            vote_range: VoteRange):
        self.vote_range: VoteRange = vote_range
        self.n_agents: int = 0

    def validate_bet(self,
            bet: WeightedBet[A, S]) -> bool:
        return self.is_valid_prediction(bet.prediction)

    def is_valid_prediction(self,
            prediction: List[float]) -> bool:
        min_pred: float = self.min_possible_vote_total()
        max_pred: float = self.max_possible_vote_total()
        prediction_at_t: float
        for prediction_at_t in prediction:
            if not min_pred <= prediction_at_t <= max_pred:
                return False
        return True


class SumVotingConfig(Generic[A, S], VotingConfigurationBase[A, S]):
    def __init__(self, vote_range: VoteRange):
        super().__init__(vote_range)

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
        return self.n_agents * self.vote_range.maxVote()

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
        return self.n_agents * self.vote_range.minVote()

    def aggregate_votes(self,
            votes: List[float]) -> float:
        return sum(votes)


class ClassicalVotingConfig(Generic[A, S], SumVotingConfig[A, S]):
    def __init__(self):
        super().__init__(BinaryVoteRange())


class DirectCardinalVotingConfig(Generic[A, S], SumVotingConfig[A, S]):
    def __init__(self):
        super().__init__(UnboundedVoteRange())


class RecommendedVoting(Generic[A, S], VotingConfigurationBase[A, S]):
    def __init__(self):
        super().__init__(ZeroToTenVoteRange())

    def max_possible_vote_total(self, dt: int = 1) -> float:
        return self.n_agents * sqrt(self.vote_range.maxVote())

    def min_possible_vote_total(self, dt: int = 1) -> float:
        return self.n_agents * sqrt(self.vote_range.minVote())

    def aggregate_votes(self,
            votes: List[float]) -> float:
        return sum(map(sqrt, votes))

