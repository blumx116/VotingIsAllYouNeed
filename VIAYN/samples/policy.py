from abc import abstractmethod
from typing import Generic, List, Dict, Optional, Tuple

import numpy as np
from numpy.random import default_rng, Generator

from VIAYN.project_types import PolicyConfiguration, A, B, S, WeightedBet
from VIAYN.utils import weighted_mean_of_bets, argmax, dict_argmax
from VIAYN.DiscreteDistribution import  DiscreteDistribution


DevolvedDiscreteDistribution: DiscreteDistribution = DiscreteDistribution.from_weighted_vals([-np.inf], [1],
                                                                                             np.random.default_rng())


class GreedyPolicyConfiguration(Generic[A, S], PolicyConfiguration[A, float, S]):
    """
    Selects the action with the highest weighted mean predicted value.
    Predicted value is summed evenly across all timesteps.
    Those values are then weighted by the standard weights of people who cast them
    """

    def __init__(self, rng: Optional[Generator] = None):
        self.rng: Generator = rng if rng is not None else np.random.default_rng()

    def validate_bet(self, bet: WeightedBet[A, S]) -> bool:
        # TODO: this should probably do some actual validation
        # but I don't want it to be duplicated with other validations
        return True

    def aggregate_bets(self,
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, float]:
        """
        Sums across all timesteps and then takes the weighted mean for each action

        Parameters
        ----------
        predictions: Dict[A, List[WeightedBet[A, S]]]
            the bets cast for each action

        Returns
        -------
        expectations: Dict[A, float]
            the weighted mean of predictiosn for each action
        """
        return {action: sum(weighted_mean_of_bets(bets)) for action, bets in predictions.items()}

    def select_action(self,
            aggregate_bets: Dict[A, float]) -> A:
        """
        Deterministically take the action with the highest expected value
        
        Parameters
        ----------
        aggregate_bets: Dict[A, float]
            the weighted mean of predictiosn for each action

        Returns
        -------
        action: A
            the selected action
        """
        result: Optional[A] = dict_argmax(aggregate_bets, self.rng)
        assert result is not None
        return result

    def action_probabilities(self,
            aggregate_bets: Dict[A, float]) -> Dict[A, float]:
        """
        Deterministic policy, so the action with the highest expectation has 100%
        TODO: does this work with ties?

        Parameters
        ----------
        aggregate_bets: Dict[A, float]
            the weighted mean of predictions for each action
        
        Returns
        -------
        probabilities: Dict[A, float]
            the probability of each action being selected
        """

        chosen_action: A = self.select_action(aggregate_bets)
        return {action: (1. if action == chosen_action else 0.) for action in aggregate_bets.keys()}


class ThompsonPolicyBase(Generic[A, B, S], PolicyConfiguration[A, B, S]):
    """
    Contains base infrastructure that is used for both ThompsonPayoutConfiguration and
    ThompsonPayoutConfiguration2
    """
    
    def __init__(self,
            random_seed: Optional[int] = None):
        """

        Parameters
        ----------
        random_seed: Optional[int]
            random_seed for the generator used for sampling
        """
        self.rng: Generator = default_rng(random_seed)

    def validate_bet(self, bet: WeightedBet[A, S]) -> bool:
        # TODO: this should probably do some actual validation
        # but I don't want it to be duplicated with other validations
        return True

    @abstractmethod
    def aggregate_bets(self,
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, B]:
        ...

    @abstractmethod
    def select_action(self,
            aggregate_bets: Dict[A, B]) -> A:
        ...

    def action_probabilities(self,
            aggregate_bets: Dict[A, B]) -> Dict[A, float]:
        """
        Estimates the probabilities of taking each action by Monte Carlo sampling
        because a closed form solution is very computationally expensive
        
        Parameters
        ----------
        aggregate_bets: Dict[A, B]
            aggregated information about bet distributions. Type of B is different for each
            version of ThompsonConfiguration

        Returns
        -------
        probabilities: Dict[A, float]
            the monte-carlo estimate of how often each action is chosen
        """
        
        n_samples: int = 10000  # TODO : should this scale up with the number of actions??
        counts: Dict[A, float] = {action: 0. for action in aggregate_bets}
        for _ in range(n_samples):
            choice: A = self.select_action(aggregate_bets)
            counts[choice] += 1
        for action in counts:
            counts[action] /= n_samples
        return counts


class ThompsonPolicyConfiguration(Generic[A, S], ThompsonPolicyBase[A, List[DiscreteDistribution], S]):
    """
        NOTE: because we made the decision to support the possibility of different bets @ each timestep
        this code enacts a slightly different version of Thompson Sampling than described in the whitepaper
        Namely, for each action it generates a list of discrete distributions according to the distribution
        @ that timestep. It then samples from that distribution @ each timestep and sums the samples.
        These sums of samples are used to compare actions
        In principle, this likely reduces the variance of the sampling process and is computationally more
        expensive than what we could use if we know that all of the bets are the same for all timesteps
        in a single timestepsA

        Alternate Explanation:
        For each action & each timestep, constructs a DiscreteDistribution from the predictions.
        Then samples from each of those discrete distributions to obtain a single prediction for the
        value of that action at that timestep.

        All timesteps are then summed together, and the action with the highest total value is chosen.

    """

    def __init__(self,
            random_seed: Optional[int] = None):
        super().__init__(random_seed)

    def aggregate_bets(self,
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, List[DiscreteDistribution]]:
        result: Dict[A, List[DiscreteDistribution]] = {action: [] for action in predictions.keys()}
        action: A
        for action in predictions:
            bets: List[WeightedBet[A, S]] = predictions[action]
            bet: WeightedBet[A, S]
            aggregator: List[List[Tuple[float, float]]] = []
            # out er list iterates by prediction timestep
            # inner list is list of (prediction, bet amount) pairs for each bet
            for bet in bets:
                t: int
                prediction: float
                bet_amount: float
                for t, (prediction, bet_amount) in enumerate(zip(bet.prediction, bet.bet)):
                    while t >= len(aggregator):
                        # should be equivalent to using 'if'
                        aggregator.append([])
                    aggregator[t].append((prediction, bet_amount * bet.money))

            # instantiate distributions for this action from aggregator
            distributions: List[DiscreteDistribution] = []
            for t in range(len(aggregator)):
                preds: List[float] = [prediction for prediction, _ in aggregator[t]]
                bet_amounts: List[float] = [bet_amount for _, bet_amount in aggregator[t]]
                distrib: DiscreteDistribution
                if sum(bet_amounts) > 0:
                    distrib = DiscreteDistribution.from_weighted_vals(
                        vals=preds, weights=bet_amounts, random_seed=self.rng)
                else:
                    distrib = DevolvedDiscreteDistribution
                distributions.append(distrib)
            result[action] = distributions
        return result

    def select_action(self,
            aggregate_bets: Dict[A, List[DiscreteDistribution]]) -> A:
        assert len(np.unique(list(map(len, aggregate_bets.values())))) == 1
        # check that all of the lists have equal length
        def sample_sum(distributions: List[DiscreteDistribution]) -> float:
            return sum([distribution.sample() for distribution in distributions])
        scores: Dict[A, float] = {action: sample_sum(dists) for action, dists in aggregate_bets.items()}
        chosen: Optional[A] = dict_argmax(scores, self.rng)
        assert chosen is not None
        return chosen


class ThompsonPolicyConfiguration2(Generic[A, S], ThompsonPolicyBase[A, DiscreteDistribution, S]):
    """
    ThompsonSampling as I originally planned it to be.
    Assumes all values in a given WeightedBet's bet are the same

    First sums each prediction over all timesteps, then uses that, combined with the bet amount
    to generate a single DiscreteDistribution per action.
    Then, samples the discrete distribution for each action & selects the action with the highest
    sampled value.
    """

    def __init__(self,
            random_seed: Optional[int] = None):
        self.rng: Generator = default_rng(random_seed)

    def aggregate_bets(self,
                   predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, DiscreteDistribution]:
        result: Dict[A, DiscreteDistribution] = {}
        action: A
        for action in predictions:
            if sum(map(lambda bet: bet.bet[0], predictions[action])) == 0.:
                result[action] = DevolvedDiscreteDistribution
            else:
                result[action] = DiscreteDistribution.from_weighted_vals(
                    vals=[sum(bet.prediction) for bet in predictions[action]],
                    weights=[bet.bet[0] * bet.money for bet in predictions[action]],
                    random_seed=self.rng)
        return result

    def select_action(self,
            aggregate_bets: Dict[A, DiscreteDistribution]) -> A:
        scores: Dict[A, float] = {action: dist.sample() for action, dist in aggregate_bets.items()}
        chosen: Optional[A] = dict_argmax(scores, self.rng)
        assert chosen is not None
        return chosen
