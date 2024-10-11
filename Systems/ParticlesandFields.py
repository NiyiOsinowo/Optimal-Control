'''This repository contains a template for Particles and Fields'''
import numpy as np
from dataclasses import *
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Any, List, Tuple
import scipy.integrate as integrate
import os
project_path= os.path.dirname(os.path.abspath(os.curdir))
import sys
sys.path.insert(0, project_path+ '/Tools')
from EnforceTyping import EnforceClassTyping, enforce_method_typing
from MDPFramework import MDPEnvironment

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


@dataclass(kw_only=True)
class ParticleInField(MDPEnvironment):
  """
  A class used to represent a particle in a Field

  Attributes
  ----------
  field: ClassicalField
    The field that the particle is in 
  particle: ClassicalParticle
    The particle that is in the field
  target: np.ndarray 
    The target position of the particle
  distance_weight: float 
    The weight of the distance between the particle and the target
  energy_weight: float 
    The weight of the energy of the particle
  terminal_signal_weight: float 
    The weight of the terminal signal of the particle
  current_time: float 
    The current time of the system

  Methods
  ------- 
  transition_model(self, state: State, action: Any)-> State: 
    Represents the  
  reward_model(self, state: State, action: Any, next_state: State, terminal_signal: bool)-> float:
    Represents the reward of the system
  is_terminal_condition(self, state: State)-> bool: 
    Represents the terminal condition of the system
  transition_step(self, state: State, action: Any)-> tuple[float, State, bool]: 
    Represents the transition step of the system
  sample_trajectory(self, runtime: float)-> list[State]: 
    Samples a trajectory of the system
  trajectory_value(self, trajectory: list[State])-> float: 
    Represents the value of the trajectory
  reset(self): 
    Resets the system

  """
  field: Field
  particle: ClassicalParticle
  target_position: np.ndarray # m
  proximity_weight: float= 1.0
  energy_weight: float= 1.0
  terminal_signal_weight: float= 1000.0
  current_time: float = 0.0# s
  state_dims: tuple= None
  action_dims: tuple= None
  
  @dataclass 
  class State(EnforceClassTyping):
    '''This class represents the state of the Agent with its position, velocity and the Field Strength if experiences at its position. 
    These are parameters the agent is able to observe, they uniquely define the state of the agent.'''
    position: np.ndarray # m
    velocity: np.ndarray #m/s
    
    def vector(self):
      return np.concatenate([self.position, self.velocity])
    
  initial_state: State = None
  current_state: State = None

  def __post_init__(self):
    assert self.target_position.shape == self.field.dimensionality, "The target has wrong dimensions"
    if self.initial_state is None:
        self.initial_state= self.random_state()
    self.current_state= self.initial_state
    self.action_dims= self.field.dimensionality
    self.state_dims= self.field.dimensionality * 2

  @enforce_method_typing
  def state_dynamics(self, state: np.ndarray, time: float, control: np.ndarray):
    """
    Compute the dynamics of the particle's state.

    Parameters:
    state (np.ndarray): The current state of the particle [x, y, vx, vy].
    time (float): The current time.
    control_force (np.ndarray): The external control force applied to the particle.

    Returns:
    np.ndarray: The derivative of the state [vx, vy, ax, ay].
    """
    velocity = state[2:]
    acceleration = (self.particle.charge * self.field.dynamics(state[:2]) + control) / self.particle.mass
    return np.concatenate((velocity, acceleration))
  
  @enforce_method_typing
  def state_transition_model(self, state: State, control: np.ndarray= np.array([0.0, 0.0]), time_interval:float= 0.1)-> State:
    """
    Computes the next state of the system after applying a constant force for a given time interval.

    Args:
        state (State): The current state of the system.
        action (np.ndarray, optional): The constant force to apply. Defaults to [0.0, 0.0].
        time_interval (float, optional): The time interval to apply the force for. Defaults to 0.1.

    Returns:
        State: The next state of the system.
    """
    time_span = [self.current_time, self.current_time + time_interval]
    next_state_vector = integrate.odeint(self.state_dynamics, state.vector(), time_span, args=(control,))[-1]
    next_position = next_state_vector[:2]
    next_velocity = next_state_vector[2:]
    return self.State(next_position, next_velocity)
  
  @enforce_method_typing
  def reward_model(self, state: State, control: np.ndarray, next_state: State, terminal_signal: bool) -> float:
    """
    Computes the reward for the agent given a state transition.

    The reward is a weighted sum of three components:
    1. Distance gained towards the target
    2. Energy consumed during the transition
    3. Terminal signal (e.g. reaching the target or running out of energy)

    Args:
        state: The current state of the agent
        action: The action taken by the agent
        next_state: The resulting state after taking the action
        terminal_signal: A boolean indicating whether the episode has terminated

    Returns:
        float: The reward value
    """
    distance_gained = np.linalg.norm(state.position - self.target_position) - np.linalg.norm(next_state.position - self.target_position)
    energy_consumed = (control[0]**2 + control[1]**2)/2
    reward = (
        self.proximity_weight * distance_gained
        # - self.energy_weight * energy_consumed
        - self.terminal_signal_weight * int(terminal_signal)
        )
    return reward
  
  @enforce_method_typing
  def is_terminal_condition(self, state: State) -> bool:
    """
    Checks if the state is outside the viable learning region of the environment.

    Args:
        state (State): The current state of the environment.

    Returns:
        bool: True if the state is outside the viable learning region, False otherwise.
    """
    x_bound = -10.0 <= state.position[0] <= 10.0
    y_bound = -10.0 <= state.position[1] <= 10.0
    velocity_bound = np.linalg.norm(state.velocity) < 10.0

    return not (x_bound and y_bound and velocity_bound)
  
  @enforce_method_typing
  def transition_step(
      self, 
      state: State, 
      action: np.ndarray = np.array([0.0, 0.0]), 
      time_interval: float = 0.1
  ) -> Tuple[State, float, bool]:
      """
      Simulates a single time step of the environment.

      Args:
          state (State): The current state of the environment. Defaults to current_state.
          action (np.ndarray): The action to take in the environment. Defaults to [0.0, 0.0].
          time_interval (float): The time interval for the simulation. Defaults to 0.1.

      Returns:
          Tuple[State, float, bool]: A tuple containing the next state, the reward, and a terminal signal.
      """
      next_state = self.state_transition_model(state, action, time_interval=time_interval)
      terminal_signal = self.is_terminal_condition(next_state)
      reward = self.reward_model(state, action, next_state, terminal_signal)
      return next_state, reward, terminal_signal

  @enforce_method_typing
  def sample_trajectory(
      self, 
      runtime: float, 
      initial_state: State = None, 
      n_steps: int = 200
  ) -> Tuple[List[Any], List[np.ndarray], List[float]]:
      """
      Generates a random state trajectory within the viable learning region.

      Args:
      - runtime (float): The total time for the trajectory in seconds.
      - initial_state (State): The initial state of the trajectory. Defaults to current_state.
      - n_steps (int): The number of steps in the trajectory. Defaults to 200.

      Returns:
      - A tuple containing the state trajectory, action trajectory, and time points.
      """
      time_interval = runtime/n_steps
      if initial_state == None:
         state = self.current_state
      else:
         state = initial_state
      time= 0.0
      state_trajectory = []
      time_points = np.linspace(time, runtime, n_steps)
      return_value= 0.0

      for t in time_points:
          state_trajectory.append(state)
          state, reward, _ = self.transition_step(state, time_interval=time_interval)
          return_value += reward
      return state_trajectory, return_value, time_points

  @enforce_method_typing
  def plot_trajectory(self, state_trajectory: list, time: np.ndarray) -> None:
      """
      Plot the trajectory of states over time.

      Args:
      - state_trajectory: A list of States representing the trajectory.
      - time: A list of time points corresponding to each state in the trajectory.

      Returns:
      - None (plots the trajectory)
      """
      positions = np.array([state.position for state in state_trajectory])
      velocities = np.array([state.velocity for state in state_trajectory])

      plt.figure(figsize=(8, 8))
      plt.plot(positions[:, 0], positions[:, 1], label='Trajectory')
      plt.scatter(positions[0, 0], positions[0, 1], c='k', marker='o', label='Start')
      plt.scatter(positions[-1, 0], positions[-1, 1], c='r', marker='*', label='End')
      xmax= max(abs(positions[:, 0]))
      ymax= max(abs(positions[:, 1]))
      true_max= max((xmax, ymax))
      plt.xlim(-2*true_max, 2*true_max)
      plt.ylim(-2*true_max, 2*true_max)
      plt.grid(True)
      plt.legend()
      plt.show()

      plt.figure(figsize=(8, 8))
      plt.plot( time, velocities[:, 0], label='Velocity x')
      plt.plot( time, velocities[:, 1], label='Velocity y')
      plt.grid(True)
      plt.legend()
      plt.show()

  def random_state(self) -> State:
      """
      Generates a random state within the viable learning region.

      Returns:
          State: A random state within the viable learning region
      """
      position = np.random.uniform(-10.0, 10.0, size=self.field.dimensionality)
      velocity = np.zeros_like(position)
      return self.State(position, velocity)

