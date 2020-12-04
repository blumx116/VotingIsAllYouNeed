# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 20:23:15
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-04 00:14:55
from VIAYN.project_types import Agent, Environment,Dict
from VIAYN.project_types import PolicyConfiguration, PayoutConfiguration, VotingConfiguration
from typing import Any

class AgentFactory:
    """
    Creates different types of Agents Based on dictionary specified

    List of acceptable configs:
    'type': 'constant','random'
    'vote': float (default = 0.0 or highest vote)
    'prediction': ActionBet(bet = [0.0], prediction = [0.0])
    'seed': float
    'N': integer specifying T (len of prediction)
    'totalVotesBound': [float func,float func] max total votes

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

class PayoutConfigFactory:
    """
    Creates different types of Payout Configs Based on dictionary specified

    List of acceptable configs:
    TBD

    Parameters
    ----------
    spec: Dict
        Information to initialize Payout Config

    Returns
    -------
    PayoutConfiguration
        created payout config based on spec
    """

    def create(self,spec: Dict[str,Any]) -> PayoutConfiguration:
        return PayoutConfiguration()


class PolicyConfigFactory:
    """
    Creates different types of Policy Configs Based on dictionary specified

    List of acceptable configs:
    TBD

    Parameters
    ----------
    spec: Dict
        Information to initialize Policy Config

    Returns
    -------
    PolicyConfiguration
        created policy config based on spec
    """

    def create(self,spec: Dict[str,Any]) -> PolicyConfiguration:
        return PolicyConfiguration()

class VotingConfigFactory:
    """
    Creates different types of Voting Configs Based on dictionary specified

    List of acceptable configs:
    TBD

    Parameters
    ----------
    spec: Dict
        Information to initialize Voting Config

    Returns
    -------
    VotingConfiguration
        created voting config based on spec
    """

    def create(self,spec: Dict[str,Any]) -> VotingConfiguration:
        return VotingConfiguration()