# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-11 19:03:44
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-11 19:05:03
from abc import abstractmethod
from typing import List, Tuple, Dict, Generic, Set

import numpy as np

from VIAYN.project_types import PayoutConfiguration, A, S, ActionBet, HistoryItem, Agent, WeightedBet
from VIAYN.utils import weighted_mean_of_bets


class PayoutConfigBase(Generic[A, S], PayoutConfiguration[A, S]):
    def validate_bet(self, bet: WeightedBet[A, S]) -> bool:
        return sum(bet.bet) <= 1

    def _calculate_payouts_for_action_(self,
            bets: List[WeightedBet[A, S]],
            welfare_score: float,
            t_current: int,
            t_cast_on: int) -> Dict[Agent[A, S], float]:
        t_idx: int = self._get_t_index_(t_current, t_cast_on)
        losses: Dict[Agent[A, S], float] = \
            {bet.cast_by: self.calculate_loss(
                    bet_to_evaluate=bet,
                    t_cast_on=t_cast_on,
                    t_current=t_current,
                    welfare_score=welfare_score)
                for bet in bets}
        bet_amounts: Dict[Agent[A, S], float] = \
            {bet.cast_by: bet.bet[t_idx] for bet in bets}

        payouts: List[float] = self._batch_payout_from_losses_(
            [(bet_amounts[agent], losses[agent]) for agent in losses.keys()])
        return {agent: payout for agent, payout in zip(losses.keys(), payouts)}

    def calculate_all_payouts(self,
            record: HistoryItem[A, S],
            welfare_score: float,
            t_current: int) -> Dict[Agent[A, S], float]:
        relevant_bets: List[WeightedBet[A, S]] = record.predictions[record.selected_action]
        seen_agents: Set[Agent] = set()
        bet: WeightedBet[A, S]
        for bet in relevant_bets:
            agent: Agent[A, S] = bet.cast_by
            assert agent not in seen_agents
            seen_agents.add(agent)

        return self._calculate_payouts_for_action_(
            bets=relevant_bets,
            welfare_score=welfare_score,
            t_current=t_current,
            t_cast_on=record.t_enacted)

    @staticmethod
    def _get_t_index_(
            t_current: int,
            t_cast_on: int) -> int:
        delta_t: int = t_current - t_cast_on
        assert delta_t > 0
        # we use 0-based indexing
        # every prediction should be checked by NEXT
        # timestep at the earliest
        return delta_t - 1

    @abstractmethod
    def _batch_payout_from_losses_(self,
            weighted_losses: List[Tuple[float, float]]) -> List[float]:
        """

        Parameters
        ----------
        weighted_losses: List[Tuple[float, float]]
            [(bet_amount, loss)] each pair corresponds to an agent
        Returns
        -------
        payout: List[float]
            payout corresponding to each element of the tuple
        """
        ...

    @staticmethod
    def _squared_loss_(
            bet_to_evaluate: WeightedBet,
            t_cast_on: int,  # timestep info let us look up in the array
            t_current: int,  # which prediction is for this timestep
            welfare_score: float) -> float:
        prediction: List[float] = bet_to_evaluate.prediction
        t_idx: int = PayoutConfigBase._get_t_index_(t_current, t_cast_on)
        assert t_idx < len(prediction)
        return (prediction[t_idx] - welfare_score) ** 2


class SimplePayoutConfig(Generic[A, S], PayoutConfigBase[A, S]):
    def calculate_loss(self,
            bet_to_evaluate: WeightedBet,
            t_cast_on: int,  # timestep info let us look up in the array
            t_current: int,  # which prediction is for this timestep
            welfare_score: float) -> float:
        return self._squared_loss_(
            bet_to_evaluate=bet_to_evaluate,
            t_cast_on=t_cast_on, t_current=t_current,
            welfare_score=welfare_score)

    def calculate_payout_from_loss(self,
            bet_amount_to_evaluate: float,
            loss_to_evaluate: float,
            all_losses: List[Tuple[float, float]],  # [(weight, loss)]
            t_cast_on: int,  # timestep info lets us discount by timestep
            t_current: int,
            action_bet_on: A,
            action_selected: A) -> float:
        if action_bet_on == action_selected:
            return bet_amount_to_evaluate * \
                   (float(np.max(all_losses)) - loss_to_evaluate)
        else:
            return 0.

    def _batch_payout_from_losses_(self,
            weighted_losses: List[Tuple[float, float]]) -> List[float]:
        """

        Parameters
        ----------
        weighted_losses: List[Tuple[float, float]]
            [(bet_amount, loss)] each pair corresponds to an agent
        Returns
        -------
        payout: List[float]
            payout corresponding to each element of the tuple
        """
        max_loss: float = max([loss for _, loss in weighted_losses])
        return [bet_amount * (max_loss - loss) for bet_amount, loss in weighted_losses]


class SuggestedPayoutConfig(Generic[A, S], PayoutConfigBase[A, S]):
    def calculate_loss(self,
            bet_to_evaluate: WeightedBet,
            t_cast_on: int,  # timestep info let us look up in the array
            t_current: int,  # which prediction is for this timestep
            welfare_score: float) -> float:
        return self._squared_loss_(
            bet_to_evaluate=bet_to_evaluate,
            t_cast_on=t_cast_on, t_current=t_current,
            welfare_score=welfare_score)

    def calculate_payout_from_loss(self,
            bet_amount_to_evaluate: float,
            loss_to_evaluate: float,
            all_losses: List[Tuple[float, float]],  # [(weight, loss)]
            t_cast_on: int,  # timestep info lets us discount by timestep
            t_current: int,
            action_bet_on: A,
            action_selected: A) -> float:
        if action_bet_on != action_selected:
            return 0.
        mean: float = self._mean_loss_(all_losses)
        maximum: float = self._max_loss_(all_losses)
        return bet_amount_to_evaluate *\
            (maximum - loss_to_evaluate) / (maximum - mean)

    @staticmethod
    def _mean_loss_(
            losses: List[Tuple[float, float]]) -> float:
        # TODO: in here and below, there is a probably-spurious cast
        # to float to appease the type-checker. see #13
        return float(np.average(
            a=[loss for _, loss in losses],
            weights=[weight for weight, _ in losses]))

    @staticmethod
    def _max_loss_(
            losses: List[Tuple[float, float]]) -> float:
        return float(np.max([loss for _, loss in losses]))

    def _batch_payout_from_losses_(self,
            weighted_losses: List[Tuple[float, float]]) -> List[float]:
        """

        Parameters
        ----------
        weighted_losses: List[Tuple[float, float]]
            [(bet_amount, loss)] each pair corresponds to an agent
        Returns
        -------
        payout: List[float]
            payout corresponding to each element of the tuple
        """
        mean: float = self._mean_loss_(weighted_losses)
        maximum: float = self._max_loss_(weighted_losses)
        return [bet_amount * (maximum - loss) / (maximum - mean)
                for bet_amount, loss in weighted_losses]
