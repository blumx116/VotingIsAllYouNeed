# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:18:03 2020
@author: suhail
"""

import sys
from os.path import dirname, abspath
path = abspath(dirname(dirname(abspath(__file__))))
sys.path.insert(0, path)
import pytest
import VIAYN.project_types as project_types # pylint: disable=import-error
import VIAYN.samples.factory as factory
import VIAYN.samples.vote_ranges as vote_range
import numpy as np

@pytest.fixture
def constant_agent_config():
    return [
    (project_types.ActionBet([0.],[5.]) , 5.),
    (project_types.ActionBet([0.],[4.]) , 5.),
    (project_types.ActionBet([0.5],[1.]) , 5.),
    (project_types.ActionBet([0.5],[0.]), 0.),
    (project_types.ActionBet([0.5],[2.]), 2.),
    (project_types.ActionBet([0.5],[2.]), 1.),
    (project_types.ActionBet([0.5],[7.]), 10.),
    (project_types.ActionBet([0.,0.43,0.1215,0.8,1.],[4.,1,4,1,4]), 9.),
    (project_types.ActionBet([0.,0.1215,0.8,1.],[4.,4,1,4]), 1.),
    (project_types.ActionBet([0.,0.43,0.1215,0.8],[4.,1,4,1]), 2.),
    (project_types.ActionBet([0.,0.43,0.8,1.],[4.,1,4,4]), 7.),
    (project_types.ActionBet([0.1215,0.8,1.],[1,1,4]), 5.),
    (project_types.ActionBet([0.,1.],[1,4]), 9.5)
]

@pytest.fixture
def random_agent_config():
    return  [
    (5.,0),
    (5., 123),    
    (0., 1231131),
    (1., 1235234),
    (2., 5124),
    (8., 82384),
    (10., 0),
]