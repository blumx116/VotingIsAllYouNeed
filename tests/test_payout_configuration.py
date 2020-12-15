# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-09 20:10:07
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-12 17:22:51
from typing import Dict, List
from tests.conftest import floatIsEqual
import pytest
import VIAYN.project_types as P
import VIAYN.samples.factory as fac
import VIAYN.samples.vote_ranges as vote_range
import numpy as np
from VIAYN.project_types import PayoutConfiguration, HistoryItem, A, S, ActionBet, Agent, WeightedBet
from VIAYN.samples.factory import PayoutConfigEnum as PCE

# def payout_config_isomorphism(
#         config: PayoutConfiguration,
#         record: HistoryItem,
#         welfare_score: float,
#         t_current: int):
#     t_cast_on: int = record.t_enacted
#     selected_action: A
#     action: A
#     total_payouts: Dict[Agent[A, S], float] = {}
#     for action in record.available_actions():
#         abets: Dict[Agent[A, S], WeightedBet] = \
#             {wb.cast_by: wb.to_action_bet() for wb in record.predictions[action]}
#         losses: Dict[Agent[A, S], float] = \
#             { agent: config.calculate_loss(abet, t_cast_on, t_current, welfare_score)
#                 for agent, abet in abets.items()}
#         loss_list: List[float] = [loss for agent, loss in losses.items()]
#         payouts: Dict[Agent[A, S], float] = \
#             {agent : config.calculate_payout_from_loss(
#                 loss_to_evaluate=loss, all_losses=loss_list,
#                 t_cast_on=t_cast_on, t_current=t_current,
#                 action_bet_on=action, action_selected=selected_action)
#             for agent, loss in losses.items()}
#         for agent in payouts:
#             if agent not in total_payouts:
#                 total_payouts[agent] = 0.
#             total_payouts[agent] += payouts[agent]

#     other_payouts = config.calculate_all_payouts(record, welfare_score, t_current)
#     assert len(other_payouts) == len(total_payouts)
#     for agent in total_payouts:
#         assert agent in other_payouts
#         assert total_payouts[agent] == other_payouts[agent]
       
@pytest.mark.parametrize("enum,pred,t1,t0,R,expected", [
    (PCE.simple,[],1,0,0,None), # What happens here?
    (PCE.simple,[3],1,0,1,4),
    (PCE.simple,[0,3],2,0,1,4),
    (PCE.simple,[10,10,10,3,10,10,10],4,0,1,4),
    (PCE.simple,[10,10,10,3,10,10,10],3,0,1,81),
    (PCE.suggested,[],1,0,0,None), # What happens here?
    (PCE.suggested,[3],1,0,1,4),
    (PCE.suggested,[0,3],2,0,1,4),
    (PCE.suggested,[10,10,10,3,10,10,10],4,0,1,4),
    (PCE.suggested,[10,10,10,3,10,10,10],3,0,1,81),
])
def test_payout_config_calculate_loss(
    enum,pred,t1,t0,R,expected,
    gen_payout_conf, gen_weighted_bet
):
    pf: P.PayoutConfiguration = gen_payout_conf(
        enum
    )
    wb: P.WeightedBet  = gen_weighted_bet(pred,pred)
    assert(floatIsEqual(pf.calculate_loss(wb,t0,t1,R),expected))

@pytest.mark.parametrize("enum,bet,t1,t0,loss,allLs,aj,ai,expected", [
    (
        PCE.simple,1,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        10*1
    ),
    (
        PCE.simple,5,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        10*5
    ),
    (
        PCE.simple,5,1,0,10,[(5,0),(4,1),(0.01,10)],1,1,
        0
    ),
    (
        PCE.simple,5,1,0,5,[(5,0),(4,5),(0.01,10)],1,1,
        5*5
    ),
    (
        PCE.simple,5,1,0,5,[(5,5),(4,5),(0.01,5)],1,1,
        0
    ),
    (
        PCE.simple,5,1,0,5,[(5,5)],1,1,
        0
    ),
    (
        PCE.suggested,1,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        1*(10/(10-((5*0+4*1+10*0.01)/(5+4+0.01))))
    ),
    (
        PCE.suggested,5,1,0,0,[(5,0),(4,1),(0.01,10)],1,1,
        5*(10/(10-((5*0+4*1+10*0.01)/(5+4+0.01))))
    ),
    (
        PCE.suggested,5,1,0,10,[(5,0),(4,1),(0.01,10)],1,1,
        0
    ),
    (
        PCE.suggested,5,1,0,5,[(5,0),(4,5),(0.01,10)],1,1,
        5*(5/(10-((5*0+4*5+10*0.01)/(5+4+0.01))))
    ),
    (
        PCE.suggested,5,1,0,5,[(5,5),(4,5),(0.01,5)],1,1,
        0
    ),
    (
        PCE.suggested,5,1,0,5,[(5,5)],1,1,
        0
    ),
])
def test_payout_config_calculate_payout_from_loss(
    enum,bet,t1,t0,loss,allLs,aj,ai,expected,
    gen_payout_conf, gen_weighted_bet
):
    pf: P.PayoutConfiguration = gen_payout_conf(
        enum
    )
    assert(
        floatIsEqual(
            pf.calculate_payout_from_loss(
                bet,loss,allLs,t0,t1,aj,ai
            ),
            expected
        )
    )

@pytest.mark.parametrize("enum,welfare_score,selectedA, t_current,weightedBets,expected", [
    (
        PCE.simple,
        5.5,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 10.0,'A2':0.}
    ),
    (
        PCE.simple,
        5.5,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':2.8}
    ),
    (
        PCE.simple,
        7,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':2.0}
    ),
    (
        PCE.simple,
        9,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.25}
    ),
    (
        PCE.simple,
        8,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.35}
    ),
    (
        PCE.simple,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.}
    ),
    (
        PCE.simple,
        5.5,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 10.0,'A2':0., 'A3': 5.6}
    ),
    (
        PCE.simple,
        5.5,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 96.6,'A2':67.2, 'A3': 0.0}
    ),
    (
        PCE.simple,
        7,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.,'A2':2.0,'A3':2.025}
    ),
    (
        PCE.simple,
        9,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.8125,'A2':0.41250000000000003, 'A3': 0.0}
    ),
    (
        PCE.simple,
        8,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 12.0,'A2':2.75, 'A3': 0.0}
    ),
    (
        PCE.simple,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 1.024,'A2':1.024,'A3': 0.0}
    ),
    (
        PCE.suggested,
        3,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.}
    ),
    (
        PCE.suggested,
        4,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.5}
    ),
    (
        PCE.suggested,
        4.5,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.7}
    ),
    (
        PCE.suggested,
        7.5,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.3,'A2':0.}
    ),
    (
        PCE.suggested,
        0,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.3 ,'A2':0.}
    ),
    (
        PCE.suggested,
        1,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
        ],
        {'A1': 0.,'A2':0.}
    ),
    (
        PCE.suggested,
        5.5,
        'a1',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 1.1538461538461542 ,'A2':0., 'A3':0.6461538461538464}
    ),
    (
        PCE.suggested,
        5.5,
        'a2',1,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.30076923076923073,'A2':0.20923076923076922, 'A3': 0.0}
    ),
    (
        PCE.suggested,
        7,
        'a2',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.,'A2':0.49689440993788814,'A3':0.5031055900621118}
    ),
    (
        PCE.suggested,
        9,
        'a1',3,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.2653061224489796,'A2':0.1346938775510204, 'A3': 0.0}
    ),
    (
        PCE.suggested,
        8,
        'a1',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.3254237288135593,'A2':0.07457627118644068, 'A3': 0.0}
    ),
    (
        PCE.suggested,
        4,
        'a2',2,
        [
            ([0.5,0.25,0.25],[5,4,6],'a1',5,'A1'),
            ([0.2,0.4,0.4],[3,7,9],'a2',3,'A2'),
            ([0.9,0.05,0.05],[1,5,11],'a1',3,'A2'),
            ([0.3,0.4,0.3],[1,7,10],'a2',5,'A1'),
            ([0.4,0.1,0.1],[3,0,5.5],'a1',15,'A3'),
            ([0.01,0.4,0.3],[24,7.4,8.5],'a2',15,'A3'),
        ],
        {'A1': 0.6,'A2': 0.6,'A3': 0.0}
    ),
])
def test_payout_config_calculate_all_payouts(
    enum, # factory
    welfare_score,
    selectedA, t_current, # history
    weightedBets,
    expected,
    gen_payout_conf, gen_weighted_bet,gen_history_item # fixtures
):
    pf: P.PayoutConfiguration = gen_payout_conf(
        enum
    )
    predsDict: Dict[A, List[WeightedBet[A, S]]] = {}
    for bet,pred,action,money,castby in weightedBets:
        if (action not in predsDict.keys()):
            predsDict[action] = []
        predsDict[action].append(
            gen_weighted_bet(
                bet,pred,action,money,castby
            )
        )
    record: HistoryItem[A,S] = gen_history_item(
        selectedA,
        predsDict,
        0
    )
    payouts = pf.calculate_all_payouts(
        record,
        welfare_score,
        t_current,
    )
    for agent in payouts:
        assert(
            floatIsEqual(
                payouts[agent],
                expected[agent]
            )
        )