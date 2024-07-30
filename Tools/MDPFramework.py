'''This repository contains a detailed implementation of the MDP Framework class'''
from dataclasses import *
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Any
@dataclass(kw_only=True)
class MDPEnvironment(ABC):  

  class State:
      pass
  
  initial_state: State 
  current_state: State 

  @abstractmethod
  def transition_model(self, state: State, action: Any)-> State:
      ...

  @abstractmethod
  def reward_model(self, state: State, action: Any, next_state: State, terminal_signal: bool)-> float:
      '''This is a scalar performance metric.'''
      ...

  @abstractmethod
  def is_terminal_condition(self, state: State)-> bool:
      ...

  @abstractmethod
  def transition_step(self, state: State, action: Any)-> tuple[float, State, bool]:
      ...

  @abstractmethod
  def sample_trajectory(self, runtime: float)-> list[State]:
      ...

  @abstractmethod
  def trajectory_value(self, trajectory: list[State])-> float:
      ...

  @abstractmethod
  def reset(self):
      ...

@dataclass(kw_only=True)
class MDPController(ABC):
  environment: MDPEnvironment
  policy: Callable

  @abstractmethod
  def act(self, observation: np.ndarray)-> np.ndarray:
      ...
  @abstractmethod
  def observe(self)-> np.ndarray:
      ...