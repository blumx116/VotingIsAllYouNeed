# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-11 19:03:44
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-11 19:05:03
from abc import abstractmethod
from typing import List, Tuple, Dict, Generic, Set, Callable

import numpy as np

from VIAYN.project_types import PayoutConfiguration, A, S, ActionBet, HistoryItem, Agent, WeightedBet, Weighted
from VIAYN.utils import map_vals, weighted_mean

"""
This code handles identifying how much money agents should receive based on the accuracy of their previously 
placed bets.

PayoutConfigBase is an optional base class that contains a lot of useful patterns that the payouts generally follow,
but it is not mandatory.

In general, it should be possible to reconstruct the behaviour of calculate_all_payouts from the behaviour of
calculate_payouts_from_loss and calculate_losses in the same way for all betting configurations.
"""


class PayoutConfigBase(Generic[A, S], PayoutConfiguration[A, S]):
    """
    Optional base class that other PayoutConfigurations can extend from.
    Present so that we can avoid having duplicate code.
    """

    def __init__(self,
            upper_bound_fn: Callable[[List[Weighted]], float]):
        self.upper_bound: Callable[[List[Weighted]], float] = upper_bound_fn

    def validate_bet(self, bet: WeightedBet[A, S]) -> bool:
        return sum(bet.bet) <= 1

    @staticmethod
    def advantage(
            loss: float,
            baseline: float) -> float:
        """
        How much less loss you got than the baseline
        
        Parameters
        ----------
        loss: float
            the loss to compare to the baseline (lower is better)
        baseline: float
            baseline to compare against  
            if you do worse than this, return 0

        Returns
        -------
        advantage: float
            how much better the loss was than the baseline  
            (higher is better)
        """
        
        # TODO: maybe vectorize this?
        # TODO: oh, the things I do for typechecker
        return float(np.max((0, baseline - loss)))

    @staticmethod
    def max_loss(
            weighted_losses: List[Weighted]) -> float:
        return float(np.max([w.val for w in weighted_losses]))

    def _calculate_payouts_for_action_(self,
            bets: List[WeightedBet[A, S]],
            welfare_score: float,
            t_current: int,
            t_cast_on: int) -> Dict[Agent[A, S], float]:
        """
        Calculates how much payout agents should receive for their bets on a
        given action (calculated agnostic to whether or not the action was selected)

        Parameters
        ----------
        bets: List[WeightedBet[A, S]]
            the bets that were cast on this action
        welfare_score: float
            the total votes that the bets should be compared against
        t_current: int
            the current timestep that the welfare_score was collected at
            used because bets may be multi-timestep
        t_cast_on: int
            the timestep that the bets were cast on.
            also used because bets may be multi-timestep

        Returns
        -------
        payouts: Dict[Agent[A, S], float]
            the amount of payout to each agent
        """
        t_idx: int = self._get_t_index_(t_current, t_cast_on)
        # the index of the prediction & bet corresponding to the current timestep
        
        assert len(np.unique([len(bet.prediction) for bet in bets])) == 1
        if len(bets) == 0:
            return {}
        if len(bets[0].prediction) <= t_idx:
            # no payout if bets do not apply to current timestep
            return {}

        losses: Dict[Agent[A, S], float] = \
            {bet.cast_by: self.calculate_loss(
                    bet_to_evaluate=bet,
                    t_cast_on=t_cast_on,
                    t_current=t_current,
                    welfare_score=welfare_score)
                for bet in bets}
        # calculate losses for each agent
        bet_amounts: Dict[Agent[A, S], float] = \
            {bet.cast_by: bet.weight()[t_idx] for bet in bets}
        # TODO: rename this (and subsequent variables) to WEIGHT
        # bet amount is confusing
        # extract bet amounts

        if len(np.unique(list(losses.values()))) == 1: # everyone has the same loss
            return bet_amounts  # just give everyone their money back

        payouts: List[float] = self._batch_payout_from_losses_(
            [Weighted(bet_amounts[agent], losses[agent]) for agent in losses.keys()])
        return {agent: payout for agent, payout in zip(losses.keys(), payouts)}

    def calculate_all_payouts(self,
            record: HistoryItem[A, S],
            welfare_score: float,
            t_current: int) -> Dict[Agent[A, S], float]:
        relevant_bets: List[WeightedBet[A, S]] = record.predictions[record.selected_action]
        # only bets that correspond to the action selected get any money
        # (this is fair because no money was withdrawn for the other action's bets)

        seen_agents: Set[Agent] = set()
        bet: WeightedBet[A, S]
        for bet in relevant_bets:
            agent: Agent[A, S] = bet.cast_by
            assert agent not in seen_agents
            seen_agents.add(agent)
        # check that there is at most one bet per agent
        # TODO: we could possibly do away with this assumption, but it would need
        # a significant refactor

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
            weighted_losses: List[Weighted]) -> List[float]:
        """

        Parameters
        ----------
        weighted_losses: List[Weighted]]
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
        # squared error between the prediction for this timestep
        # and the welfare_score
        prediction: List[float] = bet_to_evaluate.prediction
        t_idx: int = PayoutConfigBase._get_t_index_(t_current, t_cast_on)
        assert t_idx < len(prediction)
        return (prediction[t_idx] - welfare_score) ** 2


class SimplePayoutConfig(Generic[A, S], PayoutConfigBase[A, S]):
    def __init__(self,
            upper_bound_fn: Callable[[List[Weighted]], float]):
        super().__init__(upper_bound_fn)

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
            bet_amount_to_evaluate: float,  # weight (money * bet_amount)
            loss_to_evaluate: float,
            all_losses: List[Weighted],  # [(weight, loss)]
            t_cast_on: int,  # timestep info lets us discount by timestep
            t_current: int,
            action_bet_on: A,
            action_selected: A) -> float:
        # loss = max_error - min_error
        # payout scaled by bet amount
        if action_bet_on == action_selected:
            if len(np.unique([w.val for w in all_losses])) == 1:
                return bet_amount_to_evaluate
            return bet_amount_to_evaluate * \
                self.advantage(loss_to_evaluate,self.upper_bound(all_losses))
        else:
            # no payout for non-selected actions
            return 0.

    def _batch_payout_from_losses_(self,
            weighted_losses: List[Weighted]) -> List[float]:
        """

        Parameters
        ----------
        weighted_losses: List[Weighted]
            [(bet_amount, loss)] each pair corresponds to an agent
        
        Returns
        -------
        payout: List[float]
            payout corresponding to each element of the tuple
        """
        max_loss: float = self.upper_bound(weighted_losses)

        # same as calculate_payout_from_loss except max computation is cached
        return [loss.weight * PayoutConfigBase.advantage(loss.val, max_loss) for loss in weighted_losses]


class SuggestedPayoutConfig(Generic[A, S], PayoutConfigBase[A, S]):
    """
        loss = squared error between prediction and welfare_score
        payout = weight_i * (max_loss - loss_i) / (max_loss - weighted_mean_loss)
        where weighted mean is calculated by weights
    """

    def __init__(self,
            upper_bound_fn: Callable[[List[Weighted]], float]):
        super().__init__(upper_bound_fn)

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
            bet_amount_to_evaluate: float,  # WEIGHT (name is confusing)
            loss_to_evaluate: float,
            all_losses: List[Weighted],  # [(weight, loss)]
            t_cast_on: int,  # timestep info lets us discount by timestep
            t_current: int,
            action_bet_on: A,
            action_selected: A) -> float:
        if action_bet_on != action_selected:
            return 0. # no payout for non-selected actions

        if len(np.unique([w.val for w in all_losses])) == 1:
            return bet_amount_to_evaluate  # TODO: duplicate with SimpleConfig calculate_payout_from_loss

        maximum: float = self.upper_bound(all_losses)
        all_losses = map_vals(all_losses, lambda loss: PayoutConfigBase.advantage(loss, maximum))
        mean: float = weighted_mean(all_losses)

        return bet_amount_to_evaluate *\
            self.advantage(loss_to_evaluate, maximum) / mean

    def _batch_payout_from_losses_(self,
            weighted_losses: List[Weighted]) -> List[float]:
        """

        Parameters
        ----------
        weighted_losses: List[Weighted]
            [(bet_amount, loss)] each pair corresponds to an agent
        
        Returns
        -------
        payout: List[float]
            payout corresponding to each element of the tuple
        """
        maximum: float = self.upper_bound(weighted_losses)
        weighted_gains: List[Weighted] = map_vals(weighted_losses,
            fn=lambda loss: PayoutConfigBase.advantage(loss, maximum))
        mean: float = weighted_mean(weighted_gains)
        # TODO: introduces potential division by 0 error if one person contains more than 95%
        # of the vote & also does the best

        # same as calculate losses except that mean & max are cached
        return [(gain.weight * gain.val) / mean for gain in weighted_gains]
