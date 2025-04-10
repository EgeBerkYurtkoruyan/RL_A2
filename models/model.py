import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from torchinfo import summary

## We will define here the policy network which is a represented by a set of parameters in this case weights in a nueral network

class PolicyNet(nn.Module):
    def __init__(self,state_size,action_size,l1_units=64,l2_units=128):
        super(PolicyNet,self).__init__()
        self.fc1 = nn.Linear(state_size,l1_units)
        self.fc2 = nn.Linear(l1_units,l2_units)
        self.fc3 = nn.Linear(l2_units,action_size)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
    
if __name__ == '__main__':
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    print(f"Each state is defined by  {state_size} variables")
    action_size = env.action_space.n
    q_model= PolicyNet(state_size, action_size,64,128)
    print("Model was correctly initialized")
    summary(q_model, input_size=(1, state_size))







