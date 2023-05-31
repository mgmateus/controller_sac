import random
import numpy as np


from stage_diferencial import Robot

class Env:
    def __init__(self, state_dim, action_dim, max_steps):
        self.__robot = Robot()
        self.__observation_space = np.zeros(shape=(state_dim,))
        self.__action_dim = action_dim

        self.__n_steps = 0
        self.__max_steps = max_steps
        self.__past_distance = None

        self.__targets = [(2.8, 3.0), (2.0, -0.8), (0.0, 5.0), (-2.0, 0.0), (-2.0, 0.0), (4.0, 5.0), (-4.0, 6.0), (-2.0, 6.0)]
        self.__target_reset = 0
    
    @property
    def robot(self):
        return self.__robot    
    
    
    def reset(self, eps= 0, factor= 50):
        '''
        if eps > 1 and eps % factor == 0:
            self.__robot.reset_position()
            if self.__targets:
                self.__robot.target = self.__targets[0]
                self.__targets.pop(0)
            else:
                self.__robot.target = (float("{:.2f}".format(random.uniform(-4.5, 4.5))), float("{:.2f}".format(random.uniform(-1.5, 7.5))))
        '''
        
        self.__robot.reset_position()
        self.__n_steps = 0
        _ = self.__robot.get_state(np.zeros(shape=(self.__action_dim,)))
        
        return self.__observation_space
    

    def set_reward(self, done):
        if done:
            self.__robot.reset_position()
            if self.__targets:
                if self.__target_reset == 200:
                    self.__robot.target = self.__targets[0]
                    self.__targets.pop(0)
                    self.__target_reset = 0
                else:
                    self.__target_reset += 1
            else:
                if self.__target_reset == 50:
                    self.__robot.target = (float("{:.2f}".format(random.uniform(-4.5, 4.5))), float("{:.2f}".format(random.uniform(-1.5, 7.5))))
                    self.__target_reset = 0
                else:
                    self.__target_reset += 1

                    
            _ = self.__robot.get_state(np.zeros(shape=(self.__action_dim,)))
            return 200
        
        else:
            if self.__robot.collision():
                self.__robot.reset_position()
                return -10
            else:
                r = self.__past_distance - self.__robot.distance

                return r#0 if r > 0 else -0.5
        
    def step(self, action):
        self.__past_distance = self.__robot.distance
        self.__observation_space, done = self.__robot.get_state(action)
        reward = self.set_reward(done)
        self.__n_steps += 1
        
        if self.__n_steps < self.__max_steps:
            return np.asarray(self.__observation_space), reward, done
        else:
            return np.asarray(self.__observation_space), 0.0, True
    
        