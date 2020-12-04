# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 20:23:15
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-03 20:45:15
from VIAYN.project_types import Agent, Environment,Dict
from typing import Any

class AgentFactory:
    """
    Creates different types of Agents Based on dictionary specified

    List of acceptable configs:
    'type': 'constant','random'
    'vote': float (default = 0.0 or highest vote)
    'prediction': ActionBet(bet = [0.0], prediction = [0.0])
    'seed': float

    Parameters
    ----------
    spec: Dict
        Information to initialize Agents

    Returns
    -------
    Agent
        created agent based on spec
    """

    def create(self,spec: Dict[str,Any]) -> Agent:
        return Agent()

class EnvFactory:
    """
    Creates different types of Environments Based on dictionary specified


    List of acceptable configs:
    TBD

    Parameters
    ----------
    spec: Dict
        Information to initialize Environments

    Returns
    -------
    Environment
        created environment based on spec
    """

    def create(self,spec: Dict[str,Any]) -> Environment:
        return Environment()