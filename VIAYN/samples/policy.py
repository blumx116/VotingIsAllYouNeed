from typing import Generic, List, Dict, Optional

from VIAYN.project_types import PolicyConfiguration, A, B, S, WeightedBet
from VIAYN.utils import weighted_mean, argmax
from VIAYN.DiscreteDistribution import  DiscreteDistribution


class GreedyPolicyConfiguration(Generic[A, S], PolicyConfiguration[A, float, S]):
    @staticmethod
    def aggregate_bets(
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, float]:
        return {action: sum(weighted_mean(bets)) for action, bets in predictions.items()}

    @staticmethod
    def select_action(
            aggregate_bets: Dict[A, float]) -> A:
        result: Optional[A] = argmax(list(aggregate_bets.keys()), lambda key: aggregate_bets[key])
        assert result is not None
        return result

    @staticmethod
    def action_probabilities(
            aggregate_bets: Dict[A, float]) -> Dict[A, float]:
        chosen_action: A = GreedyPolicyConfiguration.select_action(aggregate_bets)
        return {action: (1. if action == chosen_action else 0.) for action in aggregate_bets.keys()}


class ThompsonPolicyConfiguration(Generic[A, S], PolicyConfiguration[A, DiscreteDistribution, S]):
    @staticmethod
    def aggregate_bets(
            predictions: Dict[A, List[WeightedBet[A, S]]]) -> Dict[A, DiscreteDistribution]:
        ...

    @staticmethod
    def select_action(
            aggregate_bets: Dict[A, DiscreteDistribution]) -> A:
        ...

    @staticmethod
    def action_probabilities(
            aggregate_bets: Dict[A, DiscreteDistribution]) -> Dict[A, float]:
        ...
