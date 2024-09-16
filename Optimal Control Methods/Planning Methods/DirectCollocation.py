import numpy as np # linear algebra
from dataclasses import *
from abc import ABC, abstractmethod

@dataclass
class DirectCollocation(ABC):
    n_time_steps: int
    final_time: float
    initial_state: np.ndarray
    final_state: np.ndarray
    control_dims: tuple
    initial_guess: np.ndarray

    @abstractmethod
    def objective(self, flattened_trajectory: np.ndarray) -> float:
      pass
    
    @abstractmethod
    def state_dynamics(self, state: np.ndarray, time: float, control: np.ndarray):
      pass
    
    @abstractmethod
    def dynamics_constraint(self, flattened_trajectory: np.ndarray):
      pass
    
    @abstractmethod
    def boundary_constraints(self, flattened_trajectory: np.ndarray):
      pass
    
    @abstractmethod
    def solve(self):
        pass
    
    @abstractmethod
    def plot_trajectory(self, flattened_trajectory: np.ndarray= None):
      pass

