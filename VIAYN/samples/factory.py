from VIAYN.project_types import Agent, Environment,Dict
from typing import Any

class AgentFactory:
    """
    Creates different types of Agents Based on dictionary specified

    List of acceptable configs:
    'type': 'constant','random'
    'vote': float (default = 0.0 or highest vote)
    'prediction': ActionBet(bet = 0.0 or max moneys, prediction = [])

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