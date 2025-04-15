import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical
from tqdm import tqdm
from models.model import PolicyNet, ValueNet
from utils.load_file import load_config
from utils.visualize import plot_metrics


class Trainer_AdvantageActorCritic :
    def __init__(self, env_name, PolicyClass, ValueClass, config_file, results_path="results", advantage:bool=True):
        self.env_name = env_name
        self.PolicyClass = PolicyClass
        self.ValueClass = ValueClass
        self.config = config_file
        self.results_path = results_path
        self.advantage = advantage
        temp_env = gym.make(env_name)
        self.state_size = temp_env.observation_space.shape[0]
        self.action_size = temp_env.action_space.n
        temp_env.close()
     # Apply temporal difference bootstrapping
    def compute_n_step_targets(self, rewards, values, gamma, n):
        T = len(rewards)
        Q_n = []
        for t in range(T):
            G = 0
            for k in range(n):
                if t + k < T:
                    G += (gamma ** k) * rewards[t + k]
            if t + n < T:
                G += (gamma ** n) * values[t + n].item()
            Q_n.append(G)
        return torch.tensor(Q_n, dtype=torch.float32)

    def advantage_train_actor_critic(self, l_rate=None, max_steps=None, gamma=None, n=None, avg_window=None):
        if self.advantage:
            print("Training Advantage Actor-Critic")
        else:
            print("Training Actor-Critic")
        
        max_steps = max_steps or self.config["training"]["steps"]
        l_rate = l_rate or self.config["training"]["lr"]
        gamma = gamma or self.config["training"]["gamma"]
        n = n or self.config["training"]["n_steps"]
        avg_window = avg_window or self.config["training"]["avg_window"]

        #print("Initializing Policty and Value model for A2C")

        policy_net = self.PolicyClass(self.state_size, self.action_size,
                                      self.config["model"]["l1_units"],
                                      self.config["model"]["l2_units"])
        value_net = self.ValueClass(self.state_size,
                                    self.config["model"]["l1_units"],
                                    self.config["model"]["l2_units"])

        policy_optim = optim.Adam(policy_net.parameters(), lr=l_rate)
        value_optim = optim.Adam(value_net.parameters(), lr=l_rate)

        env = gym.make(self.env_name)
        total_steps = 0
        total_episodes = 0
        rewards_list = []
        steps_list = []
        avg_rewards = []

        pbar = tqdm(total=max_steps, desc="A2C Progress")

        while total_steps < max_steps:
            state = env.reset()[0]
            done = False
            log_probs = []
            values = []
            rewards = []
            ep_reward = 0
            ep_steps = 0

            while not done and total_steps < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()

                value = value_net(state_tensor)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                log_probs.append(dist.log_prob(action))
                values.append(value.squeeze(0))
                rewards.append(reward)

                state = next_state
                ep_reward += reward
                ep_steps += 1
                total_steps += 1
                pbar.update(1)

            # Convert to tensor before detach
            values_pred = torch.stack(values)              # for loss computation
            values_tensor = values_pred.detach()          
            Q_n = self.compute_n_step_targets(rewards, values_tensor, gamma, n)

            # ---- Critic Update ----
            value_loss = (Q_n - values_pred).pow(2).mean()
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

            # ---- Actor Update ----
            log_probs_tensor = torch.stack(log_probs)
            if self.advantage:
                #print("Traing A2C with advantage")
                advantages = Q_n.detach() - values_tensor
                policy_loss = -torch.sum(advantages * log_probs_tensor)
            else:
                #print("Training AC without advantage")
                policy_loss = -torch.sum(Q_n.detach() * log_probs_tensor)

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # Logging
            rewards_list.append(ep_reward)
            steps_list.append(total_steps)
            total_episodes += 1
            avg_reward = np.mean(rewards_list[-avg_window:]) if len(rewards_list) >= avg_window else np.mean(rewards_list)
            avg_rewards.append(avg_reward)

        pbar.close()
        env.close()
        return avg_rewards, steps_list, total_episodes

    def train_repetitions(self, num_iterations=1):
        rewards_reps, steps_reps, episodes_reps = [], [], []
        for it in range(num_iterations):
            print(f"Iteration {it+1}/{num_iterations}")
            rewards, steps, episodes = self.advantage_train_actor_critic()
            rewards_reps.append(rewards)
            steps_reps.append(steps)
            episodes_reps.append(episodes)
        return rewards_reps, steps_reps, episodes_reps


if __name__ == "__main__":
    env_name = "CartPole-v1"
    config_path = "config.json"
    config = load_config(config_path)

    trainer = Trainer_AdvantageActorCritic(env_name, PolicyClass=PolicyNet, ValueClass=ValueNet, config_file=config, advantage=True)
    rewards, steps, episodes = trainer.advantage_train_actor_critic()
    plot_metrics([rewards], [steps], [episodes], save_path="results", figure_name="advantage_actor_critic_single_run")
