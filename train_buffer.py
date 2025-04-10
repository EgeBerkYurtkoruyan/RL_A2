import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tqdm import tqdm



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

class Trainer:
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
        
        # Always initialize replay memory since we're always using Experience Replay
        self.memory = ReplayMemory(capacity=10000)

    def epsilon_greedy(self, q_values, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return torch.argmax(q_values).item()

    def train_qmodel(self, update_ratio=None, buffer_batch_size=None,
                    max_steps=None, l_rate=None, gamma=None, avg_window=None, epsilon=None,
                    epsilon_start=None, epsilon_end=None, epsilon_decay=None, adaptive_epsilon=None):

        # Set hyperparameters
        max_steps = max_steps if max_steps is not None else self.config["training"]["steps"]
        l_rate = l_rate if l_rate is not None else self.config["training"]["lr"]
        epsilon = epsilon if epsilon is not None else self.config["training"]["epsilon"]
        epsilon_start = epsilon_start if epsilon_start is not None else self.config["training"]["epsilon_start"]
        epsilon_end = epsilon_end if epsilon_end is not None else self.config["training"]["epsilon_end"]
        epsilon_decay = epsilon_decay if epsilon_decay is not None else self.config["training"]["epsilon_decay"]
        gamma = gamma if gamma is not None else self.config["training"]["gamma"]
        update_ratio = update_ratio if update_ratio is not None else self.config["training"]["update_ratio"]
        avg_window = avg_window if avg_window is not None else self.config["training"]["avg_window"]
        adaptive_epsilon = adaptive_epsilon if adaptive_epsilon is not None else self.config["training"]["adaptive_epsilon"]
        buffer_batch_size = buffer_batch_size if buffer_batch_size is not None else self.config["training"]["replay_n"]

        print(f"Training with Experience Replay, batch size: {buffer_batch_size}")

        if adaptive_epsilon:
            print(f"Using adaptive exploration factor: {epsilon_start}, decay: {epsilon_decay}")
            epsilon = epsilon_start
        else:
            print(f"Using constant exploration factor: {epsilon}")

        # Initialize model and optimizer
        model = self.Net(self.state_size, self.action_size, self.l1_units, self.l2_units)
        optimizer = optim.Adam(model.parameters(), lr=l_rate)
        criterion = nn.MSELoss()

        env = gym.make(self.env_name)
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
                q_values = model(state)
                action = self.epsilon_greedy(q_values, epsilon)
                next_state, reward, terminal, truncated, _ = env.step(action)
                terminal_val = terminal or truncated
                
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                # Store transition in replay memory
                self.memory.push(state, action, reward, next_state, terminal_val)

                # Update step
                total_steps += 1
                episode_reward += reward
                pbar.update(1)

                # Update the model when we have enough samples and it's time to update
                if len(self.memory) >= buffer_batch_size and total_steps % update_ratio == 0:
                    # Sample batch from memory
                    batch = self.memory.sample(buffer_batch_size)
                    states, actions, rewards, next_states, term_vals = zip(*batch)

                    states = torch.cat(states)
                    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.cat(next_states)
                    terminals = torch.tensor(term_vals, dtype=torch.float32)

                    # Compute target Q-values
                    with torch.no_grad():
                        # Get max Q-value for next state and calculate target
                        max_next_q = torch.max(model(next_states), dim=1)[0]
                        target_q_values = rewards + gamma * max_next_q * (1 - terminals)

                    # Compute current Q-values
                    current_q_values = model(states).gather(1, actions).squeeze()

                    # Compute loss and optimize model
                    loss = criterion(current_q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Move to next state
                state = next_state

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
        print(f"Training completed. Total episodes: {total_episodes}, Total steps: {total_steps}")
        print(f"Last epsilon value: {epsilon_values[-1] if epsilon_values else epsilon}")

        return avg_rewards, steps_list, total_episodes

    def train_repetitions(self, num_iterations: int=1):
        rewards_reps = []
        steps_reps = []
        episodes_reps = []
        print(f"Training model over {num_iterations} repetitions with Experience Replay")
        
        for it in range(num_iterations):
            print(f"Running iteration: {it+1}")
            avg_rewards, steps_list, total_episodes = self.train_qmodel()
            rewards_reps.append(avg_rewards)
            steps_reps.append(steps_list)
            episodes_reps.append(total_episodes)

        return rewards_reps, steps_reps, episodes_reps
    