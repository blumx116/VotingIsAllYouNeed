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


@pytest.fixture
def example_people_data():
    return [
        {
            "given_name": "Alfonsa",
            "family_name": "Ruiz",
            "title": "Senior Software Engineer",
        },
        {
            "given_name": "Sayid",
            "family_name": "Khan",
            "title": "Project Manager",
        },
    ]


@pytest.fixture
def constant_agent_config():
    return [
    (project_types.ActionBet([0.],[5.]) , 5.,10),
    (project_types.ActionBet([0.],[4.]) , 5.,1),
    (project_types.ActionBet([0.5],[1.]) , 5.,100),
    (project_types.ActionBet([0.5],[0.]), 0,20),
    (project_types.ActionBet([0.5],[2.]), 2,0),
    (project_types.ActionBet([0.5],[2.]), 1,15),
    (project_types.ActionBet([0.5],[7.]), 10,4),
    (project_types.ActionBet([0.],[4.]), 9,2),
    (project_types.ActionBet([0.],[2.]), 2,10),
    (project_types.ActionBet([1],[5.]), 5,21),
    (project_types.ActionBet([1.],[0.]), 0,6),    
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