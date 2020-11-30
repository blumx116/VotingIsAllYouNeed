# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:18:03 2020
@author: suhail
"""


import sys
from os.path import dirname, abspath
import pytest

mainRepo=abspath(dirname(dirname(abspath(__file__))))
sys.path.append(f"{mainRepo}/src")
import project_types # pylint: disable=import-error

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