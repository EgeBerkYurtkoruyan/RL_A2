
### Train Q-learn model based on 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.visualize import plot_metrics
from utils.load_file import load_config, save_metrics
from models.dqmodel import Qnet
import copy
from collections import deque



class Trainer_Naive:
    def __init__(self, env_name,Net,config_file,results_path = "results"):
        self.env_name = env_name
        self.Net  = Net
        self.results_path = results_path
        self.config = config_file
        if not self.config:
            raise ValueError("Error: Config file not loaded correctly")
        self.l1_units = self.config["model"]["l1_units"]
        self.l2_units = self.config["model"]["l2_units"]
        self.env_name = env_name
        temp_env = gym.make(self.env_name)
        self.state_size = temp_env.observation_space.shape[0]
        self.action_size = temp_env.action_space.n


    # Define epsilon greedy policy
    def epsilon_greedy(self,q_values,epsilon):
        if random.random() < epsilon:
            return random.randint(0,self.action_size-1)
        else:
            return torch.argmax(q_values).item()
    
    def train_qmodel(self,
                     update_ratio=None,
                     max_steps=None, l_rate=None, 
                     gamma=None,
                     avg_window=None,
                     epsilon=None,
                     epsilon_start=None,
                     epsilon_end=None,
                     epsilon_decay = None, adaptive_epsilon = None):
        
        # Set hyperparameters
        max_steps = max_steps if max_steps is not None else self.config["training"]["steps"]
        l_rate = l_rate if l_rate is not None else self.config["training"]["lr"]
        epsilon = epsilon if epsilon is not None else self.config["training"]["epsilon"]
        epsilon_start = epsilon if epsilon_start is not None else self.config["training"]["epsilon_start"]
        epsilon_end = epsilon_end if epsilon_end is not None else self.config["training"]["epsilon_end"]
        epsilon_decay = epsilon_decay if epsilon_decay is not None else self.config["training"]["epsilon_decay"]
        gamma = gamma if gamma is not None else self.config["training"]["gamma"]
        update_ratio = update_ratio if update_ratio is not None else self.config["training"]["update_ratio"]
        avg_window = avg_window if avg_window is not None else self.config["training"]["avg_window"]
        adaptive_epsilon = adaptive_epsilon if adaptive_epsilon is not None else self.config["training"]["adaptive_epsilon"]

        # set epsilon to strtin value
        if adaptive_epsilon:
            print(f"Training with adaptive epsilon decay: {epsilon_decay}")
            epsilon = epsilon_start
        else:
            print(f"Training with  constant exploration factor : {epsilon}")
        model = self.Net(self.state_size, self.action_size,self.l1_units,self.l2_units)
        optimizer = optim.Adam(model.parameters(), lr=l_rate)
        print(f"Training with learning rate : {l_rate}")

        print(f"Training with update to data ratio: {update_ratio}")
        
        criterion = nn.MSELoss()

        env = gym.make(self.env_name)
        total_steps = 0
        total_episodes = 0 
        rewards_list = [] 
        steps_list = []  
        avg_rewards = []
        epsilon_values = []

        
        pbar = tqdm(total=max_steps, desc="Training progress")
        while total_steps<max_steps:
            state = env.reset()[0]
            state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
            terminal_val = False
            
            episode_reward = 0
            episode_steps = 0
            while not terminal_val and total_steps<max_steps:
                q_values = model(state)
                action = self.epsilon_greedy(q_values,epsilon)
                q_value = q_values[0,action]
                next_state, reward, terminal, truncated, _ = env.step(action)
                # Store the next states in buffer

                terminal_val = terminal or truncated
                next_state = torch.tensor(next_state,dtype=torch.float32).unsqueeze(0)

                ## Perform weights update every n steps
                if total_steps % update_ratio == 0:
                    with torch.no_grad():
                        target = reward+gamma*torch.max(model(next_state))*(1-terminal_val)
                    # Compute loss
                    loss = criterion(q_value,target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Update step
                state = next_state
                total_steps+=1
                episode_reward+=reward
                episode_steps+=1
                pbar.update(1)
            # Episode resutls
            total_episodes+=1
            rewards_list.append(episode_reward)
            steps_list.append(total_steps)

            
            if adaptive_epsilon:
                epsilon_values.append(epsilon)
                epsilon = max(epsilon_end,epsilon*epsilon_decay)

            # Compute average reward over last n episodes
            avg_reward = np.mean(rewards_list[-avg_window:]) if len(rewards_list)>=avg_window else np.mean(rewards_list)
            avg_rewards.append(avg_reward)
            if total_steps%50000 == 0:
                print(f"Average reward: {avg_reward}")
        pbar.close()
        env.close()
        if adaptive_epsilon:
            print(f"Last epsilon value: {epsilon_values[-1]}")

        return avg_rewards, steps_list, total_episodes
    



    def train_repetitions(self, num_iterations: int=1):
        # Load config parameters
        rewards_reps = []
        steps_reps = []
        episodes_reps = []
        print(f"Training model over {num_iterations} repetitions")
        for it in range(num_iterations):
            print(f"Running iteration: {it+1}")
            avg_rewards, steps_list,total_episodes = self.train_qmodel()
            rewards_reps.append(avg_rewards)
            steps_reps.append(steps_list)
            episodes_reps.append(total_episodes)
        # Put reward and steps list together
        

        return rewards_reps,steps_reps,episodes_reps
    




       

if __name__ == "__main__":
    results_path = "results"
    env_name = 'CartPole-v1'
    config_path = 'config.json'
    config_file = load_config(config_path)
    iterations = config_file["training"]["iterations"]
    max_steps = config_file["training"]["steps"]
    exp_name ="dqn_naive_exp_1" 


    naivedqn = Trainer(env_name,Net = Qnet,config_file=config_file,results_path=results_path)
    # Train model over n episodes
    #avg_rewards, steps_list,  total_episodes = naivedqn.train_qmodel()
    #plot_metrics(avg_rewards,steps_list,episodes_list=total_episodes)

    ## Iterate over n repetitions
    rewards_reps, steps_reps, episodes_reps = naivedqn.train_repetitions(num_iterations = iterations)
    plot_metrics(rewards_reps,steps_reps,episodes_reps)

    # run ablation test
    #run_ablation(dft_config_file = config_file,section="training",param_name="lr", param_values=[0.001,0.005,0.01])














    











    
