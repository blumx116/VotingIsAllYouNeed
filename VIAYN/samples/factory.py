# -*- coding: utf-8 -*-
# @Author: Suhail.Alnahari
# @Date:   2020-12-03 20:23:15
# @Last Modified by:   Suhail.Alnahari
# @Last Modified time: 2020-12-04 18:49:22
from VIAYN.project_types import Agent, Environment,Dict
from VIAYN.project_types import PolicyConfiguration, PayoutConfiguration, VotingConfiguration
from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, List, Callable


VoteBoundGetter = Callable[[int], float]

@dataclass(frozen=True)
class AgentFactorySpec:
	type: str
	vote: float
	totalVotesBound: Optional[Tuple[VoteBoundGetter, VoteBoundGetter]]
	seed: Optional[float] = None
	prediction: Optional[Union[float, List[float]]] = None
	bet: Optional[Union[float, List[float]]] = None
	N: Optional[int] = None

class AgentFactory:
    """
    Creates different types of Agents Based on dictionary specified

    List of acceptable configs:
    spec: AgentFactorySpec
    	specifications to create agent with, see above

    Parameters
    ----------
    spec: Dict
        Information to initialize Agents

    Returns
    -------
    Agent
        created agent based on spec
    """

    def create(self,spec: AgentFactorySpec) -> Agent:
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
