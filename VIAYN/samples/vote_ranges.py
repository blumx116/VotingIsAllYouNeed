# -*- coding: utf-8 -*-
# @Author: Carter.Blum
# @Date:   2020-12-05 00:08:28
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-10 14:44:06
import numpy as np

from VIAYN.project_types import VoteRange


class BinaryVoteRange(VoteRange):
    
    @staticmethod
    def contains(value: float) -> bool:
        return int(value) in [0, 1]
    
    @staticmethod
    def maxVote() -> float:
        return 1.
    
    @staticmethod
    def minVote() -> float:
        return 0.


class FiveStarVoteRange(VoteRange):
    
    @staticmethod
    def contains(value: float) -> bool:
        return int(value) in [1, 2, 3, 4, 5]

    @staticmethod
    def maxVote() -> float:
        return 5.

    @staticmethod
    def minVote() -> float:
        return 1.


class ZeroToTenVoteRange(VoteRange):
    
    @staticmethod
    def contains(value: float) -> bool:
        return 0. <= value <= 10.

    @staticmethod
    def maxVote() -> float:
        return 10.
    
    @staticmethod
    def minVote() -> float:
        return 0.


class UnboundedVoteRange(VoteRange):
    @staticmethod
    def contains(value: float) -> bool:
        return True

    @staticmethod
    def maxVote() -> float:
        return float(np.inf)

    @staticmethod
    def minVote() -> float:
        return float(-np.inf)
