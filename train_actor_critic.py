import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.distributions import Categorical

from utils.load_file import load_config
from utils.visualize import plot_metrics


class ActorCriticNet(nn.Module):
    """
    A network with two heads:
      - an actor head that outputs action probabilities
      - a critic head that outputs a scalar state-value estimate.
    """
    def __init__(self, state_size, action_size, l1_units=64, l2_units=128):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(state_size, l1_units)
        self.fc2 = nn.Linear(l1_units, l2_units)
        
        # Actor: for policy (action probabilities)
        self.actor_head = nn.Linear(l2_units, action_size)
        # Critic: outputs a single state-value
        self.critic_head = nn.Linear(l2_units, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Actor head: output softmax probabilities
        logits = self.actor_head(x)
        policy = F.softmax(logits, dim=1)
        # Critic head: output a value estimate
        value = self.critic_head(x)
        return policy, value


class Trainer_ActorCritic:
    def __init__(self, env_name, Net, config_file, results_path="results"):
        """
        Initializes the trainer similar to your REINFORCE trainer.
        Parameters:
          - env_name: name of the Gymnasium environment
          - Net: network class (which must output both policy and value)
          - config_file: configuration dictionary containing training/model parameters
          - results_path: path to store any results/plots
        """
        self.env_name = env_name
        self.Net = Net
        self.config = config_file
        self.results_path = results_path

        # Create a temporary environment to get state and action dimensions.
        temp_env = gym.make(self.env_name)
        self.state_size = temp_env.observation_space.shape[0]
        self.action_size = temp_env.action_space.n
        temp_env.close()

    def train_actor_critic(self,
                           l_rate=None,
                           max_steps=None,
                           gamma=None,
                           avg_window=None,
                           n_steps=None):


        max_steps = max_steps if max_steps is not None else self.config["training"]["steps"]
        l_rate = l_rate if l_rate is not None else self.config["training"]["lr"]
        gamma = gamma if gamma is not None else self.config["training"]["gamma"]
        avg_window = avg_window if avg_window is not None else self.config["training"]["avg_window"]
        n_steps = n_steps if n_steps is not None else self.config["training"].get("n_steps", 1)

        model = self.Net(self.state_size, self.action_size, l1_units=self.config["model"]["l1_units"], l2_units=self.config["model"]["l2_units"])
        optimizer = optim.Adam(model.parameters(), lr=l_rate)

        env = gym.make(self.env_name)
        total_steps = 0
        total_episodes = 0
        rewards_list = []
        steps_list = []
        avg_rewards = []

        pbar = tqdm(total=max_steps, desc="Actor–Critic (AC) Progress")

        while total_steps < max_steps:
            # --- Collect one full episode (trace) ---
            episode_data = []
            state, _ = env.reset()
            ep_reward = 0
            done = False

            while not done and total_steps < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                policy, _ = model(state_tensor)
                dist = Categorical(policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                episode_data.append({
                    'state': state,       # state 
                    'action': action,     # action
                    'reward': reward,     # reward data 
                    'log_prob': log_prob  # log probs
                })

                state = next_state
                ep_reward += reward
                total_steps += 1
                pbar.update(1)

            total_episodes += 1
            rewards_list.append(ep_reward)
            steps_list.append(total_steps)
            current_avg = np.mean(rewards_list[-avg_window:]) if len(rewards_list) >= avg_window else np.mean(rewards_list)
            avg_rewards.append(current_avg)

            # Critic - N-step boootstrapping part 
            actor_loss = 0.0
            critic_loss = 0.0
            T = len(episode_data)

            for t in range(T):
                # We Compute n-step target at the time t
                Q_target = 0.0
                # We take the sum of the tot<l rewards for steps t to t+n-1
                for k in range(n_steps):
                    if t + k < T:
                        Q_target += (gamma ** k) * episode_data[t + k]['reward']
                    else:
                        break
                # If available, add bootstrapped value from state at t+n.
                if t + n_steps < T:
                    state_n = torch.tensor(episode_data[t + n_steps]['state'], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        _, value_n = model(state_n)
                    Q_target += (gamma ** n_steps) * value_n.item()

                Q_target_tensor = torch.tensor(Q_target, dtype=torch.float32)

                # Get value estimate V(s_t) for state at time t.
                state_t_tensor = torch.tensor(episode_data[t]['state'], dtype=torch.float32).unsqueeze(0)
                _, value_t = model(state_t_tensor)
                value_t = value_t.squeeze()

                # Critic loss: squared error between Q_target and value estimate.
                critic_loss += (Q_target_tensor - value_t) ** 2

                # Actor loss: policy gradient update using Q_target directly.
                # (No baseline subtraction is performed here.)
                actor_loss += -episode_data[t]['log_prob'] * Q_target_tensor

            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.close()
        env.close()
        return avg_rewards, steps_list, total_episodes


if __name__ == "__main__":
    results_path = "results"
    env_name = "CartPole-v1"
    config_path = "config.json"
    model_name = "Actor Critic (AC)"
    config_file = load_config(config_path)
    
    lr = config_file["training"].get("lr", 1)
    episodes = config_file["training"].get("episodes", 1)
    steps = config_file["training"].get("steps", 1)

    figure_name = f"ac_bootstrap_run_lr{lr}_ep{episodes}_steps{steps}"

    # Initialize the trainer with ActorCriticNet as the network.
    trainer = Trainer_ActorCritic(env_name, Net=ActorCriticNet, config_file=config_file, results_path=results_path)

    # Train the Actor–Critic agent (without advantage subtraction) and plot metrics.
    avg_rewards, steps_list, total_episodes = trainer.train_actor_critic()
    plot_metrics(avg_rewards, steps_list, save_path=results_path, figure_name=figure_name, model_name=model_name)
