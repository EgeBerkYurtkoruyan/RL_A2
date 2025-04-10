import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt


class Qnet(nn.Module):
    # State size = number of variables that characterize the state  (Cartpole angle, position, velocity, etc)
    def __init__(self, state_size, action_size,l1_units=16,l2_units=32):
        print(f"Initializing model with: {l1_units} and {l2_units} units ")
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(state_size,l1_units)
        self.fc2 = nn.Linear(l1_units,l2_units)
        self.fc3 = nn.Linear(l2_units,action_size)
        #self._initialize_weights()
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    print(f"Each state is defined by  {state_size} variables")
    action_size = env.action_space.n
    q_model= Qnet(state_size, action_size,16,32)
    print("Model was correctly initialized")


