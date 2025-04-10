import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import copy
from collections import deque
from tqdm import tqdm
from utils.visualize import plot_metrics
from utils.load_file import load_config, save_metrics
from models.dqmodel import Qnet

# Define Replay Memory Class
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Trainer_ddqn:
    def __init__(self, env_name, Net, config_file, results_path="results"):
        self.env_name = env_name
        self.Net = Net
        self.results_path = results_path
        self.config = config_file
        if not self.config:
            raise ValueError("Error: Config file not loaded correctly")

        self.l1_units = self.config["model"]["l1_units"]
        self.l2_units = self.config["model"]["l2_units"]
        
        temp_env = gym.make(self.env_name)
        self.state_size = temp_env.observation_space.shape[0]
        self.action_size = temp_env.action_space.n
        temp_env.close()
        
        # Initialize Replay Memory
        self.memory = ReplayMemory(capacity=100000)
    
    def epsilon_greedy(self, q_values, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return torch.argmax(q_values).item()

    def train_qmodel(self):
        # Load hyperparameters
        max_steps = self.config["training"]["steps"]
        l_rate = self.config["training"]["lr"]
        epsilon_start = self.config["training"]["epsilon_start"]
        epsilon_end = self.config["training"]["epsilon_end"]
        epsilon_decay = self.config["training"]["epsilon_decay"]
        gamma = self.config["training"]["gamma"]
        update_ratio = self.config["training"]["update_ratio"]
        avg_window = self.config["training"]["avg_window"]
        target_network_update = self.config["training"]["target_network_update"]
        buffer_batch_size = self.config["training"]["replay_n"]
        adaptive_epsilon = self.config["training"]["adaptive_epsilon"]

        # Initialize networks
        policy_net = self.Net(self.state_size, self.action_size, self.l1_units, self.l2_units)
        print("Setting target network")
        target_net = self.Net(self.state_size, self.action_size, self.l1_units, self.l2_units)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=l_rate)
        criterion = nn.MSELoss()
        env = gym.make(self.env_name)

        epsilon = epsilon_start
        total_steps = 0
        total_episodes = 0
        rewards_list = []
        steps_list = []
        avg_rewards = []
        epsilon_values = []
        pbar = tqdm(total=max_steps, desc="Training progress")

        while total_steps < max_steps:
            state = env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminal_val = False
            episode_reward = 0

            while not terminal_val and total_steps < max_steps:
                q_values = policy_net(state)
                action = self.epsilon_greedy(q_values, epsilon)
                next_state, reward, terminal, truncated, _ = env.step(action)
                terminal_val = terminal or truncated
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                # Store in replay memory
                self.memory.push(state, action, reward, next_state, terminal_val)
                
                # Train when memory is sufficient
                if len(self.memory) >= buffer_batch_size and total_steps % update_ratio == 0:
                    batch = self.memory.sample(buffer_batch_size)
                    states, actions, rewards, next_states, term_vals = zip(*batch)

                    states = torch.cat(states)
                    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.cat(next_states)
                    terminals = torch.tensor(term_vals, dtype=torch.float32)

                    # **Double DQN Update**
                    with torch.no_grad():
                        # Select best action in next state using policy_net
                        next_q_values = policy_net(next_states)
                        best_next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)

                        # Evaluate value of best action using target_net
                        next_q_target_values = target_net(next_states).gather(1, best_next_actions).squeeze()

                        # Compute target Q-values using Double DQN
                        target_q_values = rewards + gamma * next_q_target_values * (1 - terminals)

                    # Compute current Q-values using policy network
                    current_q_values = policy_net(states).gather(1, actions).squeeze()

                    # Compute loss and optimize
                    loss = criterion(current_q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Update target network periodically
                if total_steps % target_network_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
                state = next_state
                total_steps += 1
                episode_reward += reward
                pbar.update(1)
                
            total_episodes += 1
            rewards_list.append(episode_reward)
            steps_list.append(total_steps)
            
            if adaptive_epsilon:
                epsilon_values.append(epsilon)
                epsilon = max(epsilon_end, epsilon * epsilon_decay)

            avg_reward = np.mean(rewards_list[-avg_window:]) if len(rewards_list) >= avg_window else np.mean(rewards_list)
            avg_rewards.append(avg_reward)
            
            if total_episodes % 10000 == 0:
                print(f"Episode {total_episodes}, Steps: {total_steps}, Avg Reward: {avg_reward:.2f}")

        pbar.close()
        env.close()
        print(f"Training completed in {total_episodes} episodes and {total_steps} steps.")
        return avg_rewards, steps_list, total_episodes

    def train_repetitions(self, num_iterations: int = 1):
        rewards_reps = []
        steps_reps = []
        episodes_reps = []
        print(f"Training model over {num_iterations} repetitions with Experience Replay and Double DQN")

        for it in range(num_iterations):
            print(f"Running iteration: {it+1}")
            avg_rewards, steps_list,total_episodes = self.train_qmodel()
            rewards_reps.append(avg_rewards)
            steps_reps.append(steps_list)
            episodes_reps.append(total_episodes)

        return rewards_reps, steps_reps, episodes_reps

if __name__ == "__main__":
    results_path = "results"
    env_name = 'CartPole-v1'
    config_path = 'config.json'
    config_file = load_config(config_path)
    iterations = config_file["training"]["iterations"]
    
    trainer = Trainer(env_name, Net=Qnet, config_file=config_file, results_path=results_path)
    avg_rewards = trainer.train_qmodel()
    plot_metrics(avg_rewards)
