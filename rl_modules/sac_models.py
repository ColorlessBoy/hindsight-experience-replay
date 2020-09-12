import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, env_params['action'])
        self.std_layer = nn.Linear(256, env_params['action'])

    def forward(self, x, deterministic=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_layer(x)
        log_std = self.std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            return mu, 0

        normal = Normal(mu, std)
        hidden_actions = normal.rsample()
        log_prob_hidden_actions = normal.log_prob(hidden_actions).sum(axis=-1)

        actions = self.max_action * torch.tanh(hidden_actions)
        log_prob_actions = (log_prob_hidden_actions 
            - (2*(np.log(2) - hidden_actions - F.softplus(-2*hidden_actions))).sum(axis=1))

        return actions, log_prob_actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
