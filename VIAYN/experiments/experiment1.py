from typing import Any, Dict, List, Tuple, Generic, Optional

from numpy.lib.histograms import histogram
from VIAYN.project_types import (
    A, HistoryItem, S, Agent, Environment, 
    PolicyConfiguration, VotingConfiguration, PayoutConfiguration,
    SystemConfiguration, WeightedBet)
from VIAYN.samples.factory import (
    PolicyConfigEnum, PolicyConfigFactory, PolicyConfigFactorySpec,
    PayoutConfigEnum, PayoutConfigFactory, PayoutConfigFactorySpec,
    VotingConfigEnum, VotingConfigFactory, VotingConfigFactorySpec,
    AgentFactory, AgentFactorySpec, AgentsEnum,
    EnvFactory, EnvsEnum, EnvsFactorySpec)
from VIAYN.samples.vote_ranges import BinaryVoteRange
from VIAYN.samples.env import IntAction as IA
from VIAYN.train import TrainResult, train


class Agents(Generic[A, S]):
    def __init__(self):
        self.agents: Dict[str, Agent[A, S]] = {}
        self.colors: Dict[str, Optional[str]] = {}

        self.palettes:  Dict[str, List[str]] = {
            'green': ['#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824'],
            'red': ['#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f'],
            'yellow': ['#fee391', '#fec44f', '#fe9929', '#ec7014'], #'#cc4c02', '#8c2d04'],
            'blue': ['#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#034e7b']
        }

    def add(self,
            agent: Agent[A, S],
            tag: str = "agent",
            color: str = None) -> None:
        i = 1
        while tag + str(i) in self.agents:
            i += 1
        tag = tag + str(i)
        self.agents[tag] = agent

        if color is None or color not in self.palettes:
            color: str = 'blue'
        assert len(self.palettes[color]) >= 1
        rgb: str = self.palettes[color][-1]
        self.palettes[color] = self.palettes[color][:-1]
        self.colors[tag] = rgb

    def list(self) -> List[Agent[A, S]]:
        return list(self.agents.values())

a = Agents()

TOTAL_AGENTS: int = 8
a.add(
    AgentFactory.create(
        AgentFactorySpec(AgentsEnum.random,
        vote=1.,
        totalVotesBound=(lambda _: 0.99, lambda _: 1.01),
        seed=0,
        bet=0.5,
        N=1)),
    tag='great conservative',
    color='yellow')

a.add(AgentFactory.create(
    AgentFactorySpec(AgentsEnum.random,
        vote=0.,
        totalVotesBound=(lambda _: 0.9, lambda _: 1.1),
        seed=0,
        bet=0.95,
        N=1)),
    tag='good confident',
    color='green')

for i in range(6):
    a.add(AgentFactory.create(
        AgentFactorySpec(AgentsEnum.random,
            vote=0.,
            totalVotesBound=(lambda _: 0, lambda _: 8),
            seed=i,
            bet=0.5,
            N=1)),
        tag='random',
        color='red')

assert TOTAL_AGENTS == len(a.list())

environment: Environment[IA, IA] = EnvFactory.create(
    EnvsFactorySpec(EnvsEnum.default, n_actions=2))

vc: VotingConfiguration[IA, IA] = VotingConfigFactory.create(
    VotingConfigFactorySpec(
        configType=VotingConfigEnum.simple,
        voteRange=BinaryVoteRange(),
        n_agents=3))

poc: PolicyConfiguration[IA, float, IA] = PolicyConfigFactory.create(
    PolicyConfigFactorySpec(
        PolicyConfigEnum.simple))

pac: PayoutConfiguration[IA, IA] = PayoutConfigFactory.create(
    PayoutConfigFactorySpec(
        PayoutConfigEnum.suggested))

config: SystemConfiguration[IA, float, IA] = SystemConfiguration(
    vc, poc, pac)

result: TrainResult[IA, IA] = train(a.list(), environment, range(10), config, tsteps_per_episode=10)


def read_moneys(result: TrainResult[A, S]) -> Dict[Agent[A, S], List[Tuple[int, float]]]:
    balance_over_time: Dict[Agent[A, S], List[Tuple[int, float]]] = {}
    episode_history: List[HistoryItem[A, S]]
    for episode_history in result.histories:
        hist_item: HistoryItem[A, S]
        for hist_item in episode_history:
            bet: WeightedBet[A, S]
            for bet in hist_item.predictions[hist_item.selected_action]:
                agent: Agent[A, S] = bet.cast_by
                t: int = hist_item.t_enacted
                money: float = bet.money
                
                if agent not in balance_over_time:
                    balance_over_time[agent] = []
                balance_over_time[agent].append((t, money))
    return balance_over_time


balances:  Dict[Agent[IA, IA], List[Tuple[int, float]]] = read_moneys(result)

import matplotlib.pyplot as plt
for log_plot in [True, False]:
    for agent_name in a.agents:
        data = balances[a.agents[agent_name]]
        plt.plot([m for _, m in data], color=a.colors[agent_name])
    plt.legend(list(a.agents.keys()))
    ylabel: str = 'agent money'
    if log_plot:
        plt.yscale('log')
        ylabel += ' (log plot)'
    plt.ylabel(ylabel)
    plt.xlabel('timestep')
    plt.title("Does a highly confident good agent have more weight \n than a cautious great one?")
    plt.show()
print()