'''This repository contains a detailed implementation of the Reinforcement Learning MDPController class'''
from Environment import Environment
from dataclasses import *
import torch as T  
from abc import ABC, abstractmethod
@dataclass
class MDPController(ABC):
  MDPEnvironment: Environment

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
