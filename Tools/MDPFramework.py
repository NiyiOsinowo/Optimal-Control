from dataclasses import *
import numpy as np
from abc import ABC, abstractmethod
from EnforceTyping import EnforceClassTyping

@dataclass(kw_only=True)
class MDPEnvironment(ABC):  
  """Template for MDP environments."""
  
  class State:
      pass
  
  initial_state: State 
  current_state: State 

  @abstractmethod
  def state_transition_model(self, state: State, action: np.ndarray):
      ...

  @abstractmethod
  def reward_model(self, state: State, action: np.ndarray, next_state: State, terminal_signal: bool)-> float:
      '''This is a scalar performance metric.'''
      ...

  @abstractmethod
  def is_terminal_condition(self, state: State)-> bool:
      ...

  @abstractmethod
  def transition_step(self, state: State, action: np.ndarray)-> tuple[State, float, bool]:
      ...

  @abstractmethod
  def sample_trajectory(self, runtime: float)-> list[State]:
      ...

  def reset(self) -> None:
      """
      Resets the current state to the initial state and sets the current time to 0.0.
      """
      self.current_state = self.initial_state
      self.current_time = 0.0

@dataclass(kw_only=True)
class MDPController(ABC):
  environment: MDPEnvironment

  @property
  @abstractmethod
  def policy(self):
      pass

  @abstractmethod
  def act(self, observation: np.ndarray)-> np.ndarray:
      'Produces an action based on an observation of the state of the environment'
      ...
  
  @abstractmethod
  def observe(self)-> np.ndarray:
      'Produces a vector encoding the observable information of the current state of the environment'
      ...

@dataclass(kw_only=True)
class LearningAgent(MDPController, EnforceClassTyping):
  
  def learn(self):
    'Updates the policy to improve performance'
    NotImplementedError ("Subclasses must implement this method")