from abc import abstractmethod
from typing import Generic, List, Dict, Optional, Tuple

from numpy.random import RandomState

from VIAYN.project_types import PolicyConfiguration, A, B, S, WeightedBet
from VIAYN.utils import weighted_mean_of_bets, argmax, dict_argmax
from VIAYN.DiscreteDistribution import  DiscreteDistribution


class GreedyPolicyConfiguration(Generic[A, S], PolicyConfiguration[A, float, S]):
    def aggregate_bets(self,
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, float]:
        return {action: sum(weighted_mean_of_bets(bets)) for action, bets in predictions.items()}

    def select_action(self,
            aggregate_bets: Dict[A, float]) -> A:
        result: Optional[A] = dict_argmax(aggregate_bets)
        assert result is not None
        return result

    def action_probabilities(self,
            aggregate_bets: Dict[A, float]) -> Dict[A, float]:
        chosen_action: A = self.select_action(aggregate_bets)
        return {action: (1. if action == chosen_action else 0.) for action in aggregate_bets.keys()}


class ThompsonPolicyBase(Generic[A, B, S], PolicyConfiguration[A, B, S]):
    def __init__(self,
            random_seed: Optional[int] = None):
        self.random: RandomState = RandomState(random_seed)

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
        in a single timesteps
    """
    def __init__(self,
            random_seed: Optional[int] = None):
        self.random: RandomState = RandomState(random_seed)

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
                    while t >= len(result[action]):
                        # should be equivalent to using 'if'
                        aggregator.append([])
                    aggregator[t].append((prediction, bet_amount))
            distributions: List[DiscreteDistribution] = [DiscreteDistribution.from_weighted_vals(
                vals=[prediction for prediction, _ in aggregator[t]],
                weights=[bet_amount for _, bet_amount in aggregator[t]],
                random_seed=self.random)
                for t in range(len(aggregator))]
            result[action] = distributions
        return result

    def select_action(self,
            aggregate_bets: Dict[A, List[DiscreteDistribution]]) -> A:
        def sample_sum(distributions: List[DiscreteDistribution]) -> float:
            return sum([distribution.sample() for distribution in distributions])
        scores: Dict[A, float] = {action: sample_sum(dists) for action, dists in aggregate_bets.items()}
        chosen: Optional[A] = dict_argmax(scores)
        assert chosen is not None
        return chosen


class ThompsonPolicyConfiguration2(Generic[A, S], ThompsonPolicyBase[A, DiscreteDistribution, S]):
    """
    ThompsonSampling as I originally planned it to be.
    Assumes all values in a given WeightedBet's bet are the same
    """
    def __init__(self,
            random_seed: Optional[int] = None):
        self.random: RandomState = RandomState(random_seed)

    def aggregate_bets(self,
                   predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, DiscreteDistribution]:
        result: Dict[A, DiscreteDistribution] = {}
        action: A
        for action in predictions:
            result[action] = DiscreteDistribution.from_weighted_vals(
                vals=[sum(bet.prediction) for bet in predictions[action]],
                weights=[bet.bet[0] for bet in predictions[action]],
                random_seed=self.random)
        return result

    def select_action(self,
            aggregate_bets: Dict[A, DiscreteDistribution]) -> A:
        scores: Dict[A, float] = {action: dist.sample() for action, dist in aggregate_bets.items()}
        chosen: Optional[A] = dict_argmax(scores)
        assert chosen is not None
        return chosen
