'''This repository contains a template for Particles and Fields'''
import numpy as np
from dataclasses import *
from abc import ABC, abstractmethod
from EnforceTyping import EnforceClassTyping

@dataclass
class Field(ABC):
  dimensionality: tuple # Represents the dimensionality of the vector space
  @abstractmethod
  def dynamics(self, observation_position: np.ndarray)-> np.ndarray:
    # A function that returns the field vector at a given position and/or time 
    ...
  
@dataclass
class ClassicalParticle(EnforceClassTyping):
    'This class represents the electric field sources with its position in the field(Position) and the magnitude of the source(Charge)'
    mass: float # kg
    charge: float #C

@dataclass
class ClassicalField(Field):
  dimensionality: tuple

  def dynamics(self, observation_position: np.ndarray, time: float)-> np.ndarray:
    NotImplementedError ("Subclasses must implement this method")
    ...
  def potential(self, observation_position: np.ndarray, time: float)-> float:
    # A function that returns the potential at a given position and/or time in the vector field  
    ...
  def potential_difference(self, initial_position: np.ndarray, final_position: np.ndarray, time: float)-> float:
    # A function that returns the potential difference between two positions at a given time in the vector field  
    ...
  def gradient(self, observation_position: np.ndarray, time: float)-> float:
    # A function that returns the gradient at a given position and/or time in the vector field
    ...
  def curl(self, observation_position: np.ndarray, time: float)-> float:
    # A function that returns the curl at a given position and/or time in the vector field
    ...
  def divergence(self, observation_position: np.ndarray, time: float)-> float:
    # A function that returns the divergence at a given position and/or time in the vector field
    ...