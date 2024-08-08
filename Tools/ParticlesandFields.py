'''This repository contains a template for Particles and Fields'''
import numpy as np
from dataclasses import *
from abc import ABC, abstractmethod
from EnforceTyping import EnforceClassTyping

@dataclass
class Field(ABC):
  dimensionality: tuple 
  @abstractmethod
  def dynamics(self, observation_position: np.ndarray)-> np.ndarray:
    # A function that returns the field vector at a given position and/or time 
    ...
  
@dataclass
class ClassicalParticle(EnforceClassTyping):
    mass: float # kg
    charge: float #C
