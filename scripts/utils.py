import torch
import os
import random

import numpy as np
import torch.nn as nn

from cgitb import reset


class Networks:

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def weights_init_(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    '''
    def save_rewards(self, episode, reward):
        
        f = open(logPath +"/log.csv", "a", encoding='utf-8', newline='')
            
        w = csv.writer(f)
        w.writerow([episode, reward, self.save_Qvals, self.save_critic_loss, self.save_policy_loss])
        f.close() 
        print('****Rewards saved***')

    def save_state(self, episode, step, reward, state):
        if episode%10 == 0:
            self.episode = episode
        
        f = open(logPath +"/observation/"+str(self.episode)+"_state.csv", "a", encoding='utf-8', newline='')
            
        w = csv.writer(f)
        w.writerow([episode, step, reward, [state[0], state[1]], [state[2], state[3]]])
        f.close() 
        print('****Observation saved***')

    def save_test(self, test, episode, reward, time, vel):
        f = open(logPath +"/test/test.csv", "a", encoding='utf-8', newline='')
        w = csv.writer(f)
        w.writerow([test, episode, reward, time, vel[0], vel[1]])
        f.close() 
        
        print('****Test saved***')
    '''

     


class Models:
    def __init__(self, agent):
        self.__agent = agent
        self.__dir_path = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "log")
        
    @property
    def agent(self):
        return self.__agent
    
    @property
    def dir_path(self):
        return self.__dir_path
    
    @dir_path.setter
    def dir_path(self, new_dir_path) -> bool:
        self.__dir_path = new_dir_path
    
    # Save model parameters 
    def save_models(self, policy, critic, world, episode_count) -> bool:
        torch.save(policy.state_dict(), self.__dir_path + '/models/' + self.agent + '/' + world + '/' + str(episode_count) + '_policy_net.pth')
        torch.save(critic.state_dict(), self.__dir_path + '/models/' + self.agent + '/' + world + '/' + str(episode_count) + '_value_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    # Load model parameters
    def load_models(self, policy, critic, critic_target, world, episode) -> bool:
        policy.load_state_dict((torch.load(self.__dir_path + '/models/' + world + '/'+str(episode)+ '_policy_net.pth')))
        critic.load_state_dict((torch.load(self.__dir_path + '/models/' + world + '/'+str(episode)+ '_value_net.pth')))
        Networks.hard_update(critic_target, critic)
        print('***Models load***')


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    
    def sample(self, batch_size):
        
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)
    

class OUNoise(object):
    def __init__(self,  space_action_dim, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.2, decay_period=8000000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   =  space_action_dim
        self.reset()
        
    def reset(self):
        self.action = np.ones(self.action_dim) * self.mu
        
    def evolve_action(self):
        x  = self.action
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.action = x + dx
        return self.action
    
    def get_noise(self, t=0): 
        ou_action = self.evolve_action()
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_action

    
    



