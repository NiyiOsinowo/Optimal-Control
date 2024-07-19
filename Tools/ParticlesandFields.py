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
NegativeCharge= Particle(Mass=1.0, Charge= -1e-5)
PositiveCharge= Particle(Mass=10.0, Charge= 1e-5)
Sources = {"Particle": [NegativeCharge],
          "Position": [T.tensor([1.0, 1.0])]}


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

@dataclass(kw_only= True)
class ElectricField(Field):
  FieldSources: Dict

  def __call__(self, ObservationPosition: T.Tensor)->T.Tensor:
      return self.FieldStrength(ObservationPosition)
  @EnforceMethodTyping
  def FieldStrength(self, ObservationPosition: T.Tensor)->T.Tensor:
    'This function takes a list of sources and outputs the field strength experienced at any given point(s). This determines the physics of the field(an electric field in this case)'
    CoulombConstant = 8.9875e9 #N*m^2/C^2
    assert len(self.FieldSources["Particle"]) == len(self.FieldSources["Position"]), "The length of particles and fields don't match"
    for FieldSource, _ in zip(self.FieldSources["Particle"], self.FieldSources["Position"]):
      assert isinstance(FieldSource, Particle),  "The FieldSource is not a Particle"
    ElectricFieldVector = T.zeros_like(ObservationPosition)
    for FieldSource, SourcePosition in zip(self.FieldSources["Particle"], self.FieldSources["Position"]):
      PositionMatrices= T.stack([T.ones_like(ObservationPosition[0])* SourcePosition[0].item(), 
                                T.ones_like(ObservationPosition[1])* SourcePosition[1].item()])
      DisplacementVector = ObservationPosition - PositionMatrices
      DisplacementMagnitude = T.sqrt(DisplacementVector[0]**2 +DisplacementVector[1]**2)
      ElectricFieldVector += (DisplacementVector * FieldSource.Charge) / DisplacementMagnitude**2
    return CoulombConstant * ElectricFieldVector #N/C or V/m
  @EnforceMethodTyping
  def FieldPotential(self, InitialPosition: T.Tensor, FinalPosition: T.Tensor, resolution: int= 5000)-> float:
      '''This method determines the amount of work required to get one position to another in the field'''
      XInterval= (FinalPosition[0] - InitialPosition[0]) / resolution
      YInterval= (FinalPosition[1] - InitialPosition[1]) / resolution
      XPositions = [InitialPosition[0] + i * XInterval for i in range(resolution + 1)]
      YPositions = [InitialPosition[1] + i * YInterval for i in range(resolution + 1)]
      WorkDone = 0
      for i in range(resolution):
          PositionFieldStrength = self.FieldStrength(T.Tensor([XPositions[i], YPositions[i]]))
          WorkDone += - (PositionFieldStrength[0]*XInterval + PositionFieldStrength[1]*YInterval)
      return WorkDone
  @EnforceMethodTyping
  def PlotField(self):
      'This funtion plots the 2D electric vector field'
      ObservationPosition= T.meshgrid(T.linspace(self.FieldLowBound, self.FieldHighBound, 50), 
                                      T.linspace(self.FieldLowBound, self.FieldHighBound, 50))
      ObservationPosition= T.stack(ObservationPosition)
      xd, yd = self.ElectricFieldStrength(ObservationPosition)
      xd = xd / T.sqrt(xd**2 + yd**2)
      yd = yd / T.sqrt(xd**2 + yd**2)
      color_aara = T.sqrt(xd**2+ yd**2)
      fig, ax = plt.subplots(1,1)
      cp = ax.quiver(ObservationPosition[0],ObservationPosition[1],xd,yd,color_aara)
      fig.colorbar(cp)
      plt.rcParams['figure.dpi'] = 250
      plt.show()
  def Derivative(self, ObservationPosition):
    'This function returns the derivative of the field at a given point'
    CoulombConstant = 8.9875e9 #N*m^2/C^2
    assert len(self.FieldSources["Particle"]) == len(self.FieldSources["Position"]), "The length of particles and fields don't match"
    for FieldSource, _ in zip(self.FieldSources["Particle"], self.FieldSources["Position"]):
      assert isinstance(FieldSource, Particle),  "The FieldSource is not a Particle"
    ElectricFieldVector = T.zeros_like(ObservationPosition)
    for FieldSource, SourcePosition in zip(self.FieldSources["Particle"], self.FieldSources["Position"]):
      PositionMatrices= T.stack([T.ones_like(ObservationPosition[0])* SourcePosition[0].item(), 
                                T.ones_like(ObservationPosition[1])* SourcePosition[1].item()])
      DisplacementVector = ObservationPosition - PositionMatrices
      DisplacementMagnitude = T.sqrt(DisplacementVector[0]**2 +DisplacementVector[1]**2)
      ElectricFieldVector += (DisplacementVector * FieldSource.Charge) / DisplacementMagnitude**1.5
    return -CoulombConstant * ElectricFieldVector
  
class HomogenousField(Field):
    def FieldStrength(self, ObservationPosition: T.Tensor)-> T.Tensor:
        return  T.zeros((ObservationPosition.shape[0], self.Dimensions), dtype=T.float64)
    def FieldPotential(self, ObservationPosition: T.Tensor)-> float:
        return  0.0

@dataclass(kw_only= True)
class LJField(Field):
    FieldSources: list[Particle]
    FieldHighBound: float
    FieldLowBound: float
    def __call__(self, ObservationPosition: T.Tensor)->T.Tensor:
        return self.FieldStrength(ObservationPosition)
    @EnforceMethodTyping
    def FieldStrength(self, ObservationPosition: T.Tensor)->T.Tensor:
        'This function takes a list of sources and outputs the field strength experienced at any given point(s). This determines the physics of the field(an electric field in this case)'
        CoulombConstant = 8.9875e9 #N*m^2/C^2
        for FieldSource in self.FieldSources:
            if type(FieldSource) != Particle:
                raise TypeError("The input is not valid")
        assert type(ObservationPosition) == T.Tensor, "Invalid Reference point data type"
        ElectricFieldVector = T.zeros_like(ObservationPosition)
        for FieldSource in self.FieldSources:
            PositionMatrices= T.stack([T.ones_like(ObservationPosition[0])* FieldSource.Position[0].item(), 
                                            T.ones_like(ObservationPosition[1])* FieldSource.Position[1].item()])
            DisplacementVector = ObservationPosition - PositionMatrices
            DisplacementMagnitude = T.sqrt(DisplacementVector[0]**2 +DisplacementVector[1]**2)
            ElectricFieldVector += ((FieldSource.Charge) / DisplacementMagnitude**3 * DisplacementVector) - ((FieldSource.Charge) / DisplacementMagnitude**6 * DisplacementVector)
        ElectricFieldVector= CoulombConstant *ElectricFieldVector
        return ElectricFieldVector #N/C or V/m
    @EnforceMethodTyping
    def FieldPotential(self, InitialPosition: T.Tensor, FinalPosition: T.Tensor, resolution: int= 5000)-> float:
        '''This method determines the amount of work required to get one position to another in the field'''
        XInterval= (FinalPosition[0] - InitialPosition[0]) / resolution
        YInterval= (FinalPosition[1] - InitialPosition[1]) / resolution
        XPositions = [InitialPosition[0] + i * XInterval for i in range(resolution + 1)]
        YPositions = [InitialPosition[1] + i * YInterval for i in range(resolution + 1)]
        WorkDone = 0
        for i in range(resolution):
            PositionFieldStrength = self.ForceFieldStrength(T.Tensor([XPositions[i], YPositions[i]]))
            WorkDone += - (PositionFieldStrength[0]*XInterval + PositionFieldStrength[1]*YInterval)
        return WorkDone
    @EnforceMethodTyping
    def PlotField(self):
        'This funtion plots the 2D electric vector field'
        ObservationPosition= T.meshgrid(T.linspace(self.FieldLowBound, self.FieldHighBound, 40), 
                                        T.linspace(self.FieldLowBound, self.FieldHighBound, 40))
        ObservationPosition= T.stack(ObservationPosition)
        xd, yd = self.ElectricFieldStrength(ObservationPosition)
        xd = xd / T.sqrt(xd**2 + yd**2)
        yd = yd / T.sqrt(xd**2 + yd**2)
        color_aara = T.sqrt(xd**2+ yd**2)
        fig, ax = plt.subplots(1,1)
        cp = ax.quiver(ObservationPosition[0],ObservationPosition[1],xd,yd,color_aara)
        fig.colorbar(cp)
        plt.rcParams['figure.dpi'] = 250
        plt.show()
