'''This repository contains a template for Particles and Fields'''
import numpy as np
from dataclasses import *
from typing import Callable
from abc import ABC, abstractmethod
from EnforceTyping import EnforceClassTyping

@dataclass
class Field(ABC):
  Dimensionality: tuple # Represents the dimensionality of the vector space
  @abstractmethod
  def Dynamics(self, ObservationPosition: np.ndarray)-> np.ndarray:
    # A function that returns the field vector at a given position and/or time 
    ...
  
@dataclass
class ClassicalParticle(EnforceClassTyping):
    'This class represents the electric field sources with its position in the field(Position) and the magnitude of the source(Charge)'
    Mass: float # kg
    Charge: float #C

@dataclass
class ClassicalField(Field):
  Dimensionality: tuple
  @abstractmethod
  def Dynamics(self, ObservationPosition: np.ndarray, Time: float)-> np.ndarray:
    ...
  def Potential(self, ObservationPosition: np.ndarray, Time: float)-> float:
    # A function that returns the potential at a given position and/or time in the vector field  
    ...
  def PotentialDifference(self, InitialPosition: np.ndarray, FinalPosition: np.ndarray, Time: float)-> float:
    # A function that returns the potential difference between two positions at a given time in the vector field  
    ...
  def Gradient(self, ObservationPosition: np.ndarray, Time: float)-> float:
    # A function that returns the gradient at a given position and/or time in the vector field
    ...
  def Curl(self, ObservationPosition: np.ndarray, Time: float)-> float:
    # A function that returns the curl at a given position and/or time in the vector field
    ...
  def Divergence(self, ObservationPosition: np.ndarray, Time: float)-> float:
    # A function that returns the divergence at a given position and/or time in the vector field
    ...