# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-09 20:10:07
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-09 20:11:58
from typing import Dict, List

import tests.conftest
from VIAYN.project_types import PayoutConfiguration, HistoryItem, A, S, ActionBet, Agent


def payout_config_isomorphism(
        config: PayoutConfiguration,
        record: HistoryItem,
        welfare_score: float,
        t_current: int):
    t_cast_on: int = record.t_enacted
    selected_action: A
    action: A
    total_payouts: Dict[Agent[A, S], float] = {}
    for action in record.available_actions():
        abets: Dict[Agent[A, S], ActionBet] = \
            {wb.cast_by: wb.to_action_bet() for wb in record.predictions[action]}
        losses: Dict[Agent[A, S], float] = \
            { agent: config.calculate_loss(abet, t_cast_on, t_current, welfare_score)
                for agent, abet in abets.items()}
        loss_list: List[float] = [loss for agent, loss in losses.items()]
        payouts: Dict[Agent[A, S], float] = \
            {agent : config.calculate_payout_from_loss(
                loss_to_evaluate=loss, all_losses=loss_list,
                t_cast_on=t_cast_on, t_current=t_current,
                action_bet_on=action, action_selected=selected_action)
            for agent, loss in losses.items()}
        for agent in payouts:
            if agent not in total_payouts:
                total_payouts[agent] = 0.
            total_payouts[agent] += payouts[agent]

    other_payouts = config.calculate_all_payouts(record, welfare_score, t_current)
    assert len(other_payouts) == len(total_payouts)
    for agent in total_payouts:
        assert agent in other_payouts
        assert total_payouts[agent] == other_payouts[agent]