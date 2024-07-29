'''This repository contains a detailed implementation of the MDP Framework class'''
from dataclasses import *
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
@dataclass(kw_only=True)
class MDPEnvironment(ABC):  

  class State:
      pass
  InitialState: State 
  CurrentState: State 

  @abstractmethod
  def TransitionModel(self, State: State, Action)-> State:
      ...

  @abstractmethod
  def RewardModel(self, State: State, Action, NextState: State, TerminalSignal: bool)-> float:
      '''This is a scalar performance metric.'''
      ...

  @abstractmethod
  def IsTerminalCondition(self, State: State)-> bool:
      ...

  @abstractmethod
  def StateTransition(self, State: State, Action)-> tuple[float, State, bool]:
      ...

  @abstractmethod
  def SampleTrajectory(self, RunDuration: float)-> list[State]:
      ...

  @abstractmethod
  def TrajectoryValue(self, Trajectory: list[State])-> float:
      ...

  @abstractmethod
  def Reset(self):
      ...

@dataclass(kw_only=True)
class MDPController(ABC):
  MDPEnvironment: MDPEnvironment
  Policy: Callable

  @abstractmethod
  def Act(self, Observation: np.ndarray)-> np.ndarray:
      ...
  @abstractmethod
  def Observe(self)-> np.ndarray:
      ...
