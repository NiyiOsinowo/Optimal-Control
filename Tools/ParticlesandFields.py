'''This repository contains a template for Particles and Fields'''
import numpy as np
from dataclasses import *
from typing import Callable
from abc import ABC, abstractmethod
from EnforceTyping import EnforceClassTyping

@dataclass
class Particle(EnforceClassTyping):
    'This class represents the electric field sources with its position in the field(Position) and the magnitude of the source(Charge)'
    Mass: float # kg
    Charge: float #C

@dataclass
class Field(ABC):
  # Dimensionality: int
  # Dynamics: Callable
  @abstractmethod
  def FieldStrength(self, ObservationPosition: np.ndarray)-> np.ndarray:
    ...
  @abstractmethod
  def FieldPotential(self, ObservationPosition: np.ndarray)-> float:
    ...
  def PotentialDifference(self, InitialPosition: np.ndarray, FinalPosition: np.ndarray) -> float:
    ...
