'''This repository contains a detailed implementation of the Reinforcement Learning Enviroment class'''
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import *
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from functools import wraps
import os
import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
import scipy.integrate as integrate
import scipy.optimize as optimize
T.Tensor.ndim = property(lambda self: len(self.shape))
from EnforceTyping import EnforceClassTyping, EnforceFunctionTyping, EnforceMethodTyping

@dataclass
class Particle(EnforceClassTyping):
    'This class represents the electric field sources with its position in the field(Position) and the magnitude of the source(Charge)'
    Mass: float # kg
    Charge: float #C

@dataclass
class Field(ABC):
  @abstractmethod
  def FieldStrength(self, ObservationPosition: T.Tensor)-> T.Tensor:
    ...
  @abstractmethod
  def FieldPotential(self, ObservationPosition: T.Tensor)-> float:
    ...
  def PotentialDifference(self, InitialPosition: T.Tensor, FinalPosition: T.Tensor) -> float:
    ...
