import numpy as np
from dataclasses import *
import os
project_path= os.path.dirname(os.path.abspath(os.curdir))
import sys
sys.path.insert(0, project_path+ '/Tools')
from EnforceTyping import EnforceClassTyping, enforce_method_typing, enforce_function_typing
from MDPFramework import MDPEnvironment,  LearningAgent

@dataclass
class MPController(LearningAgent, EnforceClassTyping):
    
    @property
    def policy(self):
      pass
    
    @property
    def internal_model(self):
        raise NotImplementedError

    def act(self, observation: np.ndarray)-> np.ndarray:
      # TO DO: Implement
      pass

    def plan(self)-> tuple[list, list[np.ndarray]]:
      'Produces an sequence of actions and predicted states based on the observation of the current state of the environment'
      # TO DO: Implement
      pass

    def observe(self)-> np.ndarray:
      # TO DO: Implement
      pass

    def learn(self):
      # TO DO: Implement
      pass


def MPC_algorithm(environment: MDPEnvironment, agent: MPController, n_episodes: int, episode_duration: int):
    # TO DO: Implement
    pass
