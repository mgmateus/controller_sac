import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions.normal import Normal
from utils import Networks


class Actor(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, hidden_dim= 300):
        super(Actor, self).__init__()
        self.device = device
        self.fc1= nn.Linear(space_states_dim, hidden_dim).to(self.device)
        self.fc2= nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.fc3= nn.Linear(hidden_dim, space_actions_dim).to(self.device)


    def forward(self, state):
        l1 = F.relu(self.fc1(state)).to(self.device)
        l2 = F.relu(self.fc2(l1)).to(self.device)
        actions = T.tanh(self.fc3(l2))

        return actions

class Critic(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, hidden_dim= 300):
        super(Critic, self).__init__()
        self.device = device
        self.fc1= nn.Linear(space_states_dim+space_actions_dim, hidden_dim).to(self.device)
        self.fc2= nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.fc3= nn.Linear(hidden_dim, 1).to(self.device)


    def forward(self, state, action):
        x =T.cat([state, action],1).to(self.device) 
        l1 = F.relu(self.fc1(x)).to(self.device)
        l2 = F.relu(self.fc2(l1)).to(self.device)
        Q = self.fc3(l2)

        return Q
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(Networks.weights_init_)
        
    def forward(self, state, action):
        x_state_action = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear4_q1(x1)
        
        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear4_q2(x2)
        
        return x1, x2
    
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(Policy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(Networks.weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()      
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std