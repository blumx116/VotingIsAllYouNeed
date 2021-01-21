from copy import copy
from typing import List, Union, Optional

from VIAYN.project_types import Environment, Action


class IntAction(Action):
    def __init__(self, idx: int):
        self.idx: int = idx
        self.description = str(idx)

    def __hash__(self) -> int:
        return self.idx 

    def __eq__(self, other: object) -> bool:
        return type(self) == type(other) and self.idx == other.idx


class StaticEnvironment(Environment[IntAction, IntAction]):
    """
    An environment with nothing but a set of actions, denoted as integers
    The state is just the idx of the last action chosen
    """
    
    def __init__(self,
            n_actions: int):
        assert n_actions > 0
        self.action_list: List[IntAction] = [IntAction(i) for i in range(n_actions)]
        self.last_action: IntAction = self.action_list[0]

    def step(self, action: Union[int, IntAction]) -> None:
        pass

    def actions(self) -> List[IntAction]:
        return copy(self.action_list)

    def state(self) -> IntAction:
        if self.last_action is not None:
            return self.last_action
        else:
            return self.action_list[0]

    def done(self) -> bool:
        return False

    def reset(self) -> IntAction:
        return self.last_action

    def seed(self, random_seed: int = None) -> None:
        pass
