import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.model import PolicyNet
from torch.distributions import Categorical
from utils.load_file import load_config
from utils.visualize import plot_metrics




class Trainer_Reinforce:
    def __init__(self, env_name,Net,config_file,results_path = "results"):
        self.env_name = env_name
        self.Net  = Net
        self.config = config_file
        self.results_path = results_path
        temp_env = gym.make(self.env_name)
        self.state_size = temp_env.observation_space.shape[0]
        self.action_size = temp_env.action_space.n
   # Compute the return  
    def compute_trace_return(self, rewards, gamma):
        R_temp = 0
        returns = []
        for r in reversed(rewards):
            R_temp = r + gamma * R_temp
            returns.append(R_temp)
        returns.reverse()  # Fix: reverse in-place
        return returns  
    
    def train_reinforce(self,
                        l_rate = None,
                        max_steps=None,
                        gamma=None,
                        avg_window=None
                        ):
        max_steps = max_steps if max_steps is not None else self.config["training"]["steps"]
        l_rate = l_rate if l_rate is not None else self.config["training"]["lr"]
        gamma = gamma if gamma is not None else self.config["training"]["gamma"]
        avg_window = avg_window if avg_window is not None else self.config["training"]["avg_window"]

        print("Initializing model for REINFORCE")
        model = self.Net(self.state_size, self.action_size,self.config["model"]["l1_units"],self.config["model"]["l2_units"])
        optimizer = optim.Adam(model.parameters(), lr=l_rate)

        env = gym.make(self.env_name)
        total_steps = 0
        total_episodes = 0
        rewards_list = []
        steps_list = []
        avg_rewards = []

        pbar = tqdm(total=max_steps, desc="REINFORCE Progress")
        while total_steps < max_steps:
            state = env.reset()[0]
            terminal_val = False
            log_probs = []
            rewards = []
            ep_reward = 0
            ep_steps = 0
            while  not terminal_val and total_steps < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs = model(state_tensor) 
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                terminal_val = terminated or truncated
                log_probs.append(log_prob)
                rewards.append(reward)
                ep_reward += reward
                ep_steps += 1
                total_steps += 1
                state = next_state

                pbar.update(1)

            returns = self.compute_trace_return(rewards, gamma)
            returns = torch.tensor(returns)
            loss  = sum([-log_prob * R for log_prob, R in zip(log_probs, returns)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Logging
            total_episodes += 1
            rewards_list.append(ep_reward)
            steps_list.append(total_steps)

            avg_reward = np.mean(rewards_list[-avg_window:]) if len(rewards_list) >= avg_window else np.mean(rewards_list)
            avg_rewards.append(avg_reward)
        pbar.close()
        env.close()
        return avg_rewards, steps_list, total_episodes
    

    def train_repetitions(self, num_iterations:int=1):
        # Load config parameters
        rewards_reps = []
        steps_reps = []
        episodes_reps = []
        print(f"Training model over {num_iterations} repetitions")
        for it in range(num_iterations):
            print(f"Running iteration: {it+1}")
            avg_rewards, steps_list,total_episodes = self.train_reinforce()
            rewards_reps.append(avg_rewards)
            steps_reps.append(steps_list)
            episodes_reps.append(total_episodes)
        # Put reward and steps list together

        return rewards_reps,steps_reps,episodes_reps





if __name__ == "__main__":
    results_path = "results"
    env_name = "CartPole-v1"
    config_path = "config.json"
    config_file = load_config(config_path)

    # Create the trainer
    trainer = Trainer_Reinforce(env_name, Net=PolicyNet, config_file=config_file, results_path=results_path)

    # Train once instead of running multiple repetitions
    avg_rewards, steps_list, total_episodes = trainer.train_reinforce()
    plot_metrics(avg_rewards, steps_list, save_path=results_path, figure_name="reinforce_single_run")


















                




