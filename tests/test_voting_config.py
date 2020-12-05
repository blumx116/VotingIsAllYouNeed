# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-04 22:06:47
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-05 00:41:03


from conftest import pytest, project_types,factory as fac, vote_range,np


@pytest.mark.parametrize("VR", [
    vote_range.BinaryVoteRange,
    vote_range.FiveStarVoteRange,
    vote_range.ZeroToTenVoteRange
])
def test_random_simple_agent_forward_prediction(VR):
    for i in range(10):
        print("checking {VR} contains {VR.contains(i)}")

