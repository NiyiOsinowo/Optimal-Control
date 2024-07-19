'''This repository contains a detailed implementation of the Reinforcement Learning MDPController class'''
import matplotlib.pyplot as plt
from Environment import Environment
from dataclasses import *
import torch as T
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from functools import wraps
from abc import ABC, abstractmethod
@dataclass
class MDPController(ABC):
  MDPControllerEnvironment: Environment
  ControlFrequency: float
  @abstractmethod
  def Act(self, Observation: T.Tensor)-> T.Tensor:
      ...
  @abstractmethod
  def Observe(self)-> T.Tensor:
      ...
  @abstractmethod
  def Learn(self):
      'Improves  the MDPController by updating its models'
      ...
  @abstractmethod
  def LearningAlgorithm(self):
      ...
