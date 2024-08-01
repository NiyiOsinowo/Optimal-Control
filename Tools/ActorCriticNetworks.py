from dataclasses import *
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, state_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__() 
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')

        self.fc1 = T.nn.utils.parametrizations.weight_norm(nn.Linear(state_dims+n_actions, fc1_dims)) 
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = T.nn.utils.parametrizations.weight_norm(nn.Linear(fc1_dims, fc2_dims))
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.fc3 = T.nn.utils.parametrizations.weight_norm(nn.Linear(fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        x = T.relu(self.bn1(self.fc1(x)))
        x = T.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
 
class ActorNetwork(nn.Module):
    def __init__(self, learning_rate, state_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')

        self.fc1 = T.nn.utils.parametrizations.weight_norm(nn.Linear(state_dims , fc1_dims)) 
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = T.nn.utils.parametrizations.weight_norm(nn.Linear(fc1_dims, fc2_dims))
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.fc3 = T.nn.utils.parametrizations.weight_norm(nn.Linear(fc2_dims, n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.relu(self.bn1(self.fc1(state)))
        x = T.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
