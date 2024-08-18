'''This repository contains a detailed implementation of the Reinforcement Learning Enviroment class'''
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import *
import torch as T
import torch.nn as nn
import random
from collections import deque
import sys
sys.path.insert(0, '/Users/niyi/Documents/GitHub/Optimal-Control/Tools')
from OUNoise import OUNoise
from EnforceTyping import EnforceClassTyping, enforce_method_typing, enforce_function_typing
from MDPFramework import MDPEnvironment,  LearningAgent
from ActorCriticNetworks import ActorNetwork, CriticNetwork
T.Tensor.ndim = property(lambda self: len(self.shape))

@dataclass(kw_only=True)
class DDPGAgent(EnforceClassTyping, LearningAgent):

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
                 control_interval: float = 0.5,
                 control_magnitude: float= 1e-7,
                 discount_rate: float =0.99,
                 max_size: int= 1000,
                 batch_size: int= 6):
        self.environment= environment
        self.actor = ActorNetwork(observation_size, action_size, actor_layers, actor_activations, 'DDPGMainActor', actor_learning_rate)
        self.critic = CriticNetwork(observation_size, action_size, critic_layers, critic_activations, 'DDPGMainCritic', critic_learning_rate)
        self.target_actor = ActorNetwork(observation_size, action_size, actor_layers, actor_activations, 'DDPGTargetActor', actor_learning_rate)
        self.target_critic = CriticNetwork(observation_size, action_size, critic_layers, critic_activations, 'DDPGTargetCritic', critic_learning_rate)
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
        self.control_interval= self.control_interval# Action duration= conrol interval
        self.control_magnitude= control_magnitude
        self.update_network_parameters()

    @property
    def policy(self):
        return self.actor

    @enforce_method_typing
    def observe(self, state= None)-> T.Tensor:
        if state is None:
          state= self.environment.current_state   
        observation= T.Tensor(state.vector())
        return observation

    @enforce_method_typing
    def act(self, observation: T.Tensor, with_noise: bool= True):
        self.actor.eval()
        observation = observation.to(self.actor.device)
        action = self.policy(observation).to(self.actor.device)
        noise= T.tensor(self.noise(), dtype=T.float)
        noisy_action = (self.control_magnitude* (action + noise)).to(self.actor.device)
        if with_noise:
            return noisy_action.cpu().detach()
        else:
            return action.cpu().detach()

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
        
        target_actions = self.actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions) 
        q_expected = self.critic.forward(state, action)
        q_targets = reward + self.discount_rate * critic_value_ * (1 - done)

        critic_loss = nn.MSELoss()(q_expected, q_targets.detach())
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.eval()
        self.critic.eval()

        mu = self.actor.forward(state)
        Actor_loss = -self.critic.forward(state, mu)

        Actor_loss = T.mean(Actor_loss)
        self.actor.train()
        self.actor.optimizer.zero_grad()
        Actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    @enforce_method_typing
    def sample_path(self, runtime: float, n_steps: int=100):
        route= []
        route_return= 0.0
        state= self.environment.initial_state
        initial_time= 0.0
        time_points = np.linspace(initial_time, runtime, n_steps)
        for _ in time_points:
            observation= self.observe(state)
            route.append(observation)
            action= self.act(observation)
            state, reward, _= self.environment.transition_step(state, action)
            route_return += reward
        return route, route_return
    
    @enforce_method_typing
    def plot_path(self, path, runtime: float):
        Path= T.stack(Path)
        Path= Path.transpose(dim0=0, dim1=1)
        t=  T.arange(0, RunDuration)
        plt.plot(Path[0], Path[1])
        plt.plot(Path[0][0], Path[1][0], 'ko')
        plt.plot(Path[0][-1], Path[1][-1], 'r*')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.grid(True)
        plt.show()
    
    def update_network_parameters(self, SoftUpdateRate=1):
        if SoftUpdateRate is None:
            SoftUpdateRate = self.soft_update_rate

        Critic_state_dict = dict(self.critic.named_parameters())
        Actor_state_dict = dict(self.policy.named_parameters())
        TargetCritic_dict = dict(self.target_critic.named_parameters())
        TargetActor_dict = dict(self.target_actor.named_parameters())

        for name in Critic_state_dict:
            Critic_state_dict[name] = SoftUpdateRate*Critic_state_dict[name].clone() + (1-SoftUpdateRate)*TargetCritic_dict[name].clone()
        self.target_critic.load_state_dict(Critic_state_dict)

        for name in Actor_state_dict:
            Actor_state_dict[name] = SoftUpdateRate*Actor_state_dict[name].clone() + (1-SoftUpdateRate)*TargetActor_dict[name].clone()
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
        episode_return = 0
        for _ in range(episode_duration):
            observation= agent.observe()
            action = agent.act(observation) 
            new_state, reward, terminal_signal= environment.transition_step(environment.current_state, np.array(action), agent.control_interval) 
            agent.memory.append((observation, action, agent.observe(new_state), reward, int(terminal_signal)))
            agent.learn()
            episode_return += reward
            environment.current_state = new_state
        return_history.append(episode_return)
    plt.plot(return_history)
    return return_history
    