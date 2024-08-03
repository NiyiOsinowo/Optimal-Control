from dataclasses import *
import torch as T
import torch.nn as nn
import torch.optim as optim
import os

    
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, layer_activations, name, learning_rate, chkpt_dir='Data/Temp/critic_data'):
        super(CriticNetwork, self).__init__()
        if os.path.exists(chkpt_dir):
            self.checkpoint_file = os.path.join(chkpt_dir,'ddpg_'+name)
        else:
            os.makedirs(chkpt_dir)
            self.checkpoint_file = os.path.join(chkpt_dir,'ddpg_'+name)
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

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        return self.network(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.network.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.network.load_state_dict(T.load(self.checkpoint_file))
 
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, layer_activations, name, learning_rate, chkpt_dir='Data/Temp/critic_data'):
        super(ActorNetwork, self).__init__()
        if os.path.exists(chkpt_dir):
            self.checkpoint_file = os.path.join(chkpt_dir,'ddpg_'+name)
        else:
            os.makedirs(chkpt_dir)
            self.checkpoint_file = os.path.join(chkpt_dir,'ddpg_'+name)
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

    def forward(self, state):
        return self.network(state)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
