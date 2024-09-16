from dataclasses import *
import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.insert(0, '/Users/niyi/Documents/GitHub/Optimal-Control/Tools')
from EnforceTyping import enforce_method_typing
class CriticNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: tuple, layer_activations: tuple, name: str, learning_rate: float, project_path: str):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(os.path.join(project_path, 'Data', 'Temp', 'critic_data')):
            os.makedirs(os.path.join(project_path, 'Data', 'Temp', 'critic_data'))
            self.checkpoint_file = os.path.join(project_path, 'Data', 'Temp', 'critic_data', name)
        else:
            
            self.checkpoint_file = os.path.join(project_path, 'Data', 'Temp', 'critic_data', name)
        layers = []

        current_input_size = state_size+ action_size
        
        for hidden_layer_size, layer_activation_func in zip(hidden_layers, layer_activations):
            layers.append(T.nn.utils.parametrizations.weight_norm(nn.Linear(current_input_size, hidden_layer_size)) )
            layers.append(nn.LayerNorm(hidden_layer_size))
            layers.append(layer_activation_func)
            current_input_size = hidden_layer_size
        
        layers.append(T.nn.utils.parametrizations.weight_norm(nn.Linear(current_input_size, 1)))

        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    @enforce_method_typing
    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        return self.network(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.network.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            print('Checkpoint file does not exist')
        else:
            print('... loading checkpoint ...')
            self.network.load_state_dict(T.load(self.checkpoint_file))
 
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, layer_activations, name, learning_rate, project_path: str):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(os.path.join(project_path, 'Data', 'Temp', 'actor_data')):
            os.makedirs(os.path.join(project_path, 'Data', 'Temp', 'actor_data'))
            self.checkpoint_file = os.path.join(project_path, 'Data', 'Temp', 'actor_data', name)
        else:
            
            self.checkpoint_file = os.path.join(project_path, 'Data', 'Temp', 'actor_data', name)
        layers = []

        current_input_size = state_size
        
        for hidden_layer_size, layer_activation_func in zip(hidden_layers, layer_activations):
            layers.append(T.nn.utils.parametrizations.weight_norm(nn.Linear(current_input_size, hidden_layer_size)) )
            layers.append(nn.LayerNorm(hidden_layer_size))
            layers.append(layer_activation_func)
            current_input_size = hidden_layer_size
            
        layers.append(T.nn.utils.parametrizations.weight_norm(nn.Linear(current_input_size, action_size)))

        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    @enforce_method_typing
    def forward(self, state: T.Tensor):
        return self.network(state)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.network.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            print('Checkpoint file does not exist')
        else:
            print('... loading checkpoint ...')
            self.network.load_state_dict(T.load(self.checkpoint_file))
