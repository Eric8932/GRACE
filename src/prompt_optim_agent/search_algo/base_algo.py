from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np

State = TypeVar("State")
Action = TypeVar("Action")
class SearchAlgo(ABC):
    def __init__(self, 
                 task,
                 world_model, 
                 action_agent,
                 logger=None, 
                 seed=0, 
                 print_log=True,
                 test_every_step=True,
                 depth_limit = None,
                 ) -> None:
        self.task = task
        self.world_model = world_model
        self.action_agent = action_agent
        self.states = []
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.seed = seed
        self.test_every_step = test_every_step
        self.depth_limit = depth_limit

    @abstractmethod
    def search(self):
        pass

