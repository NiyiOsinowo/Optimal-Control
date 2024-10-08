'''This repository contains a detailed implementation of the Reinforcement Learning Enviroment class'''
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import *
import torch as T
import torch.nn as nn 
import random
from collections import deque
import os
project_path= os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
import sys
sys.path.insert(0, project_path+ '/Tools')
from OUNoise import OUNoise
from EnforceTyping import EnforceClassTyping, enforce_method_typing, enforce_function_typing
from MDPFramework import MDPEnvironment,  LearningAgent
from ActorCriticNetworks import ActorNetwork, CriticNetwork
T.Tensor.ndim = property(lambda self: len(self.shape))

@dataclass(kw_only=True)
class DDPGAgent(LearningAgent, EnforceClassTyping):
    def __init__(self, 
                 environment: MDPEnvironment, 
                 actor_layers: tuple,
                 critic_layers: tuple,
                 actor_activations: tuple, 
                 critic_activations: tuple,
                 observation_size: int, 
                 action_size: int,
                 actor_learning_rate: float,
                 critic_learning_rate: float,
                 soft_update_rate: float,
                 control_interval: float,
                 actor_save_path: str,
                 critic_save_path: str,
                 control_magnitude: float= 1.0,
                 discount_rate: float =0.99,
                 max_size: int= 1024,
                 batch_size: int= 64):
        self.actor_save_path= actor_save_path
        self.critic_save_path= critic_save_path
        self.environment= environment
        self.actor = ActorNetwork(observation_size, 
                                  action_size, 
                                  actor_layers, 
                                  actor_activations, 
                                  'DDPGMainActor', 
                                  actor_learning_rate, 
                                  self.actor_save_path)
        self.critic = CriticNetwork(observation_size, 
                                    action_size, 
                                    critic_layers, 
                                    critic_activations, 
                                    'DDPGMainCritic', 
                                    critic_learning_rate, 
                                    self.critic_save_path)
        self.target_actor = ActorNetwork(observation_size, 
                                         action_size, 
                                         actor_layers, 
                                         actor_activations, 
                                         'DDPGTargetActor', 
                                         actor_learning_rate, 
                                         self.actor_save_path)
        self.target_critic = CriticNetwork(observation_size, 
                                           action_size, 
                                           critic_layers, 
                                           critic_activations, 
                                           'DDPGTargetCritic', 
                                           critic_learning_rate, 
                                           self.critic_save_path)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data) 
        self.memory = deque(maxlen=max_size)
        self.soft_update_rate= soft_update_rate
        self.control_interval= control_interval
        self.discount_rate= discount_rate
        self.batch_size= batch_size
        self.noise = OUNoise(mu=np.zeros(action_size))
        self.control_interval= self.control_interval
        self.control_magnitude= control_magnitude
        self.update_network_parameters()

    @property
    def policy(self):
        return self.actor

    def observe(self, state= None)-> T.Tensor:
        if state is None:
          state= self.environment.current_state   
        observation= T.Tensor(state.vector())
        return observation

    @enforce_method_typing
    def act(self, observation: T.Tensor)-> T.Tensor:
        '''Generates a control signal based on an observation'''
        self.actor.eval()
        observation = observation.to(self.actor.device)
        action = self.policy(observation).to(self.actor.device)
        return action.cpu().detach()
    
    @enforce_method_typing
    def control_mechanism(self, control_signal: T.Tensor, with_noise: bool= True)-> np.ndarray:
        '''Generates a control force based on a control signal'''
        if with_noise:
            noise= T.tensor(self.noise(), dtype=T.float)
            noisy_control = (self.control_magnitude* (control_signal + noise)).to(self.actor.device)
            return np.array(noisy_control)
        else:
            control= (self.control_magnitude* control_signal).to(self.actor.device)
            return np.array(control)
        
    @enforce_method_typing 
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        observations, actions, next_observations, rewards, dones = zip(*batch)

        state = T.stack(observations).to(self.critic.device)
        action = T.stack(actions).to(self.critic.device)
        reward = T.tensor(rewards, dtype=T.float).unsqueeze(1).to(self.critic.device)
        new_state = T.stack(next_observations).to(self.critic.device)
        done = T.tensor(dones, dtype=T.float).unsqueeze(1).to(self.critic.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        self.critic.optimizer.zero_grad()

        with T.no_grad():
            target_actions = self.target_actor.forward(new_state)
            critic_value_ = self.target_critic.forward(new_state, target_actions) 
            q_targets = reward + self.discount_rate * critic_value_ * (1 - done)
        q_expected = self.critic.forward(state, action)

        critic_loss = nn.MSELoss()(q_expected, q_targets.detach())
        self.critic.train()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.eval()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        expected_actions = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, expected_actions)

        actor_loss = T.mean(actor_loss)
        self.actor.train()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    @enforce_method_typing
    def sample_trajectory(self, runtime: float, n_steps: int=100):
        trajectory= []
        trajectory_return= 0.0
        state= self.environment.initial_state
        initial_time= 0.0
        time_points = np.linspace(initial_time, runtime, n_steps)
        for _ in time_points:
            observation= self.observe(state)
            trajectory.append(observation)
            action= self.act(observation)
            state, reward, _= self.environment.transition_step(state, np.array(action))
            trajectory_return += reward
        return trajectory, trajectory_return
    
    @enforce_method_typing
    def plot_trajectory(self, trajectory: list):
        trajectory= T.stack(trajectory)
        trajectory= trajectory.transpose(dim0=0, dim1=1)
        px, py, vx, vy= trajectory
        plt.figure(figsize=(8, 8))
        plt.plot(px, py, label='Trajectory')
        plt.scatter(px[0], py[0], c='k', marker='o', label='Start')
        plt.scatter(px[-1], py[-1], c='r', marker='*', label='End')
        plt.xlim(-2*max(abs(px)), 2*max(abs(px)))
        plt.ylim(-2*max(abs(py)), 2*max(abs(py)))
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def update_network_parameters(self, soft_update_rate: float=None):
        if soft_update_rate is None:
            soft_update_rate = self.soft_update_rate

        Critic_state_dict = dict(self.critic.named_parameters())
        Actor_state_dict = dict(self.policy.named_parameters())
        TargetCritic_dict = dict(self.target_critic.named_parameters())
        TargetActor_dict = dict(self.target_actor.named_parameters())

        for name in Critic_state_dict:
            Critic_state_dict[name] = soft_update_rate*Critic_state_dict[name].clone() + (1-soft_update_rate)*TargetCritic_dict[name].clone()
        self.target_critic.load_state_dict(Critic_state_dict)

        for name in Actor_state_dict:
            Actor_state_dict[name] = soft_update_rate*Actor_state_dict[name].clone() + (1-soft_update_rate)*TargetActor_dict[name].clone()
        self.target_actor.load_state_dict(Actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        TargetActor_params = self.TargetActor.named_parameters()
        TargetCritic_params = self.TargetCritic.named_parameters()

        Critic_state_dict = dict(TargetCritic_params)
        Actor_state_dict = dict(TargetActor_params)
        print('\nActor Networks', tau)
        for name, param in self.Actor.named_parameters():
            print(name, T.equal(param, Actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.Critic.named_parameters():
            print(name, T.equal(param, Critic_state_dict[name]))
        input()
        """
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.policy.load_checkpoint()
        self.target_policy.load_checkpoint()
        self.value_estimator.load_checkpoint()
        self.target_value_estimator.load_checkpoint()
 
@enforce_function_typing
def DDPGAlgorithm(environment: MDPEnvironment, agent: DDPGAgent, n_episodes: int, episode_duration: int):
    return_history = []
    for _ in range(n_episodes):
        environment.reset()
        terminal_signal = False
        episode_return = 0.0
        for _ in range(episode_duration):
            current_state= environment.current_state
            observation= agent.observe(current_state)
            action = agent.act(observation) 
            control= agent.control_mechanism(action)
            new_state, reward, terminal_signal= environment.transition_step(current_state, control, agent.control_interval) 
            agent.memory.append((observation, action, agent.observe(new_state), reward, int(terminal_signal)))
            agent.learn()
            episode_return += reward
            environment.current_state = new_state
        return_history.append(episode_return)
    plt.plot(return_history)
    agent.critic.save_checkpoint()
    agent.actor.save_checkpoint()
    agent.target_critic.save_checkpoint()
    agent.target_actor.save_checkpoint()
    return return_history
    