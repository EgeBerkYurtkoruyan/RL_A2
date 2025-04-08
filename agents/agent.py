import os
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import numpy as np

import random
import matplotlib.pyplot as plt
import pandas as pd

from collections import deque
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

# Q-Network with Layer Normalization for Stability
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128]):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states), torch.LongTensor(actions), 
                torch.FloatTensor(rewards), torch.FloatTensor(next_states), 
                torch.FloatTensor(dones))
    
    def __len__(self):
        return len(self.buffer)


def create_directory(dir_name, partition):
    log_dir = f'{dir_name}/{partition}'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Training loop with Target Network and Experience Replay
def train_dqn(
        env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=500, gamma=0.99,
        epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500, batch_size=64,
        target_update_freq=50, update_interval=1, partition=None, 
        parameter_value="results", log_dir="logs"
):
    episode_rewards = []
    steps_done = 0
    epsilon = epsilon_start

    log_dir = create_directory(log_dir, partition)
    log_file = f'{log_dir}/{parameter_value}.csv'

    with open(log_file, 'w') as f:
        f.write("Episode,Reward,Epsilon,Steps\n")

        for episode in range(num_episodes):
            state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
            episode_reward = 0
            done = False

            while not done and steps_done < 1000000:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = qnet(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                if replay_buffer is not None:
                    replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                steps_done += 1

                # Epsilon Decay (Linear)
                epsilon = max(epsilon_final, epsilon - (epsilon_start - epsilon_final) / epsilon_decay)

                # Update only after 'update_interval' environment steps
                if replay_buffer is not None and len(replay_buffer) > batch_size and steps_done % update_interval == 0:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    
                    with torch.no_grad():
                        next_q_values = target_qnet(next_states).max(1)[0]
                        targets = rewards + gamma * next_q_values * (1 - dones)
                    
                    q_values = qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
                    loss = nn.MSELoss()(q_values, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Update target network periodically
                if steps_done % target_update_freq == 0:
                    target_qnet.load_state_dict(qnet.state_dict())

            episode_rewards.append(episode_reward)
            f.write(f"{episode+1},{episode_reward:.2f},{epsilon:.3f},{steps_done}\n")
            f.flush()

            if steps_done >= 1_000_000:
                break

    env.close()
    return episode_rewards

def ablation_study_learning_rates(
        parameter_name, partition, learning_rates=[5e-3, 1e-4, 5e-4],num_episodes=50000,
        hidden_sizes=[128, 128], gamma=0.99, epsilon_start=0.5,
        epsilon_final=0.000001, epsilon_decay=20000
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}

    for lr in learning_rates:
        qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet.load_state_dict(qnet.state_dict())
        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(10000)
        
        print(f"Training with learning rate: {lr}")
        log_dir = create_directory('logs', parameter_name)
        rewards = train_dqn(env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=num_episodes,
                            gamma=gamma, epsilon_start=epsilon_start, epsilon_final=epsilon_final,
                            epsilon_decay=epsilon_decay, partition=partition, parameter_value=lr,
                            log_dir=log_dir)
        results[lr] = rewards
    
    plt.figure(figsize=(10, 6))
    for lr, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=f"LR={lr}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Ablation Study: Learning Rate Effect on CartPole")
    plt.legend()

    log_dir = f'experiments/{parameter_name}'
    os.makedirs(log_dir, exist_ok=True)
    log_dir = create_directory('experiments', parameter_name)
    plt.savefig(f'{log_dir}/{partition}.png')


def ablation_study_hidden_sizes(
        parameter_name, partition, lr=1e-4, num_episodes=10000,
        hidden_sizes_list=[[128, 128, 128], [128]], gamma=0.99, epsilon_start=0.5,
        epsilon_final=0.000001, epsilon_decay=20000
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    parameter_name = 'hidden_sizes'

    results = {}

    for hidden_sizes in hidden_sizes_list:
        qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet.load_state_dict(qnet.state_dict())
        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(10000)
        
        print(f"Training with hidden sizes: {hidden_sizes}")
        log_dir = create_directory('logs', parameter_name)
        rewards = train_dqn(env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=num_episodes,
                            gamma=gamma, epsilon_start=epsilon_start, epsilon_final=epsilon_final,
                            epsilon_decay=epsilon_decay, partition=partition, parameter_value=hidden_sizes,
                            log_dir=log_dir)
        results[str(hidden_sizes)] = rewards
    
    plt.figure(figsize=(10, 6))
    for hidden_sizes, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=f"{parameter_name}={hidden_sizes}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Ablation Study: {parameter_name} Effect on CartPole")
    plt.legend()

    dir = create_directory('experiments', parameter_name)
    plt.savefig(f'{dir}/{partition}.png') 


def ablation_study_gamma(
        parameter_name, partition, lr=1e-4, num_episodes=10,
        hidden_sizes=[128,128], gammas=[0.99, 0.9, 0.7], epsilon_start=0.5,
        epsilon_final=0.000001, epsilon_decay=20000
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}

    for gamma in gammas:
        qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet.load_state_dict(qnet.state_dict())
        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(10000)
        
        print(f"Training with gamma: {gamma}")
        log_dir = create_directory('logs', parameter_name)
        rewards = train_dqn(env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=num_episodes,
                            gamma=gamma, epsilon_start=epsilon_start, epsilon_final=epsilon_final,
                            epsilon_decay=epsilon_decay, partition=partition, parameter_value=gamma,
                            log_dir=log_dir)
        results[str(gamma)] = rewards
    
    plt.figure(figsize=(10, 6))
    for gamma, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=f"{parameter_name}={gamma}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Ablation Study: {parameter_name} Effect on CartPole")
    plt.legend()

    dir = create_directory('experiments', parameter_name)
    plt.savefig(f'{dir}/{partition}.png') 


def ablation_study_final_eps(
        parameter_name, partition, lr=1e-4, num_episodes=10000,
        hidden_sizes=[128,128], gamma=0.99, epsilon_start=0.5,
        epsilon_final_list=[0.01, 0.001, 0.0001, 0.000001], epsilon_decay=20000
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}

    for epsilon_final in epsilon_final_list:
        qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet.load_state_dict(qnet.state_dict())
        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(10000)
        
        print(f"Training with gamma: {epsilon_final}")
        log_dir = create_directory('logs', parameter_name)
        rewards = train_dqn(env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=num_episodes,
                            gamma=gamma, epsilon_start=epsilon_start, epsilon_final=epsilon_final,
                            epsilon_decay=epsilon_decay, partition=partition, parameter_value=epsilon_final,
                            log_dir=log_dir)
        results[str(epsilon_final)] = rewards
    
    plt.figure(figsize=(10, 6))
    for epsilon_final, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=f"{parameter_name}={epsilon_final}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Ablation Study: {parameter_name} Effect on CartPole")
    plt.legend()

    dir = create_directory('experiments', parameter_name)
    plt.savefig(f'{dir}/{partition}.png') 


def ablation_study_update_interval(
        parameter_name, partition, lr=1e-4, num_episodes=10000,
        hidden_sizes=[128,128], gamma=0.99, epsilon_start=0.5,
        epsilon_final=0.0001, epsilon_decay=20000,
        update_intervals=[10]
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}

    for update_interval in update_intervals:
        qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet.load_state_dict(qnet.state_dict())
        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(10000)

        print(f"Training with {parameter_name}: {update_interval}")
        log_dir = create_directory('logs', parameter_name)
        rewards = train_dqn(env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=num_episodes,
                            gamma=gamma, epsilon_start=epsilon_start, epsilon_final=epsilon_final,
                            epsilon_decay=epsilon_decay, partition=partition, parameter_value=update_interval,
                            log_dir=log_dir, update_interval=update_interval)
        results[update_interval] = rewards
        
    plt.figure(figsize=(10, 6))
    for epsilon_final, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=f"{parameter_name}={epsilon_final}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Ablation Study: {parameter_name} Effect on CartPole")
    plt.legend()

    dir = create_directory('experiments', parameter_name)
    plt.savefig(f'{dir}/{partition}.png') 


def main(
    parameter_name, partition, num_episodes=10000, learning_rate=1e-4, gamma=0.99,
    epsilon_start=0.5, epsilon_final=0.001, epsilon_decay=10000, batch_size=64,
    target_update_freq=50, update_interval=1, replay_buffer_size=10000,
    hidden_sizes=[128, 128]
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}

    configurations = {
        "Naive DQN": {"use_tn": False, "use_er": False},
        "Only TN": {"use_tn": True, "use_er": False},
        "Only ER": {"use_tn": False, "use_er": True},
        "TN & ER": {"use_tn": True, "use_er": True}
    }

    for config_name, settings in configurations.items():
        print(f"Training: {config_name}")
        
        qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)
        target_qnet.load_state_dict(qnet.state_dict())

        optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)
        replay_buffer = ReplayBuffer(replay_buffer_size) if settings["use_er"] else None
        
        print(f"Training with {config_name}")
        log_dir = create_directory('logs', parameter_name)

        rewards = train_dqn(
            env, qnet, target_qnet, optimizer, replay_buffer, num_episodes=num_episodes,
            gamma=gamma, epsilon_start=epsilon_start, epsilon_final=epsilon_final,
            epsilon_decay=epsilon_decay, batch_size=batch_size,
            target_update_freq=target_update_freq if settings["use_tn"] else float('inf'),
            update_interval=update_interval, parameter_value=config_name, log_dir=log_dir,
            partition=partition
        )

        results[config_name] = rewards

    env.close()

    plt.figure(figsize=(10, 6))
    for config_name, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=config_name)
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Comparison of DQN Configurations")
    plt.legend()
    
    dir = create_directory('experiments', parameter_name)
    plt.savefig(f'{dir}/{partition}.png')


def plot_results(
        dir_name='results',
        x_axis='Steps',
        y_axis='Reward',
        title='Plot of the learning curves with different configurations',
        x_label='Environment Steps',
        y_label='Mean Reward',
        rolling_window=30,
        sigma=5,
        alpha=0.2,
        figsize=(12, 8),
        grid=True,
        legend=True,
        scatter=True,
):
    base_path = f'logs/{dir_name}'
    # Initialize a dictionary to store mean rewards for each CSV name
    csv_mean_rewards = {}
    # Iterate through each folder in the base path
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # Iterate through each CSV file in the folder
            for csv_file in os.listdir(folder_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(folder_path, csv_file)
                    # Read the CSV file
                    df = pd.read_csv(csv_path)
                    mean_rewards = df[[x_axis,y_axis]]
                    # Aggregate the mean rewards for CSVs with the same name
                    if csv_file not in csv_mean_rewards:
                        csv_mean_rewards[csv_file] = mean_rewards
                    else:
                        csv_mean_rewards[csv_file] = pd.concat([csv_mean_rewards[csv_file], mean_rewards])
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    c = 0
    plt.figure(figsize=(12, 8))
    for csv_name, mean_reward in csv_mean_rewards.items():

        mean_reward = mean_reward.sort_values(by=x_axis)
        # Create bins of 120 points of x
        mean_reward['x_bin'] = (mean_reward.index // rolling_window)

        # Calculate the mean of y for each bin
        mean_reward = mean_reward.groupby('x_bin').agg(
            {y_axis: 'mean', x_axis: 'mean'}
        ).reset_index(drop=True)
        # Scatter plot of the rewards with reduced visibility
        plt.scatter(mean_reward.Steps, mean_reward.Reward, color=colors[c], alpha=0.2)
        plt.plot(mean_reward.Steps, mean_reward.Reward, color=colors[c], alpha=0.8, label=f"{dir_name}:{csv_name[:-4]}")
        # Calculate standard deviation area
        std = mean_reward.Reward.std()
        plt.fill_between(
            mean_reward.Steps, 
            mean_reward.Reward - std,
            mean_reward.Reward + std,
            color=colors[c],
            alpha=0.2
        )
        
        # Smooth the rolling mean using a Gaussian filter
        c+=1
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H_%M_%S")

    learning_rate = 1e-4
    gamma = 0.99
    epsilon_start = 0.5
    epsilon_final = 0.001
    epsilon_decay = 10000
    batch_size = 64
    target_update_freq = 50
    update_interval = 1
    replay_buffer_size = 10000
    sizes = [128, 128]
    num_episodes = 1

    for i in range(5):
        ablation_study_learning_rates(
            'learning_rates', now, learning_rates=[5e-3, 1e-4, 5e-4],
            num_episodes=num_episodes
        )
        ablation_study_hidden_sizes(
            'hidden_sizes', now, hidden_sizes_list=[[128, 128, 128], [128]],
            num_episodes=num_episodes
        )
        ablation_study_gamma(
            'gamma', now, gammas=[0.99, 0.9, 0.7], num_episodes=1
        )
        ablation_study_final_eps(
            'epsilon_final', now, epsilon_final_list=[0.01, 0.001, 0.0001, 0.000001],
            num_episodes=num_episodes
        )
        ablation_study_update_interval(
            'update_interval', now, update_intervals=[10],
            num_episodes=num_episodes
        )

    main(
        "results", now, num_episodes=num_episodes, learning_rate=learning_rate, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_final=epsilon_final, epsilon_decay=epsilon_decay,
        batch_size=batch_size, target_update_freq=target_update_freq, update_interval=update_interval,
        replay_buffer_size=replay_buffer_size, hidden_sizes=sizes
    )

    plot_results(dir_name="learning_rates")