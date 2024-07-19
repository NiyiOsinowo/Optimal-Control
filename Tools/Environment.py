'''This repository contains a detailed implementation of the Reinforcement Learning Enviroment class'''

from dataclasses import *
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from functools import wraps
from abc import ABC, abstractmethod

@dataclass(kw_only=True)
class Environment(ABC):  

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
      