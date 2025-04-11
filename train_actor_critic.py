import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
from models.model import PolicyNet
# from models.value_network import ValueNet
from utils.load_file import load_config
from utils.visualize import plot_metrics

class ValueNet(nn.Module):
    def __init__(self, state_size, l1_units=64, l2_units=128):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_size, l1_units)
        self.fc2 = nn.Linear(l1_units, l2_units)
        self.fc3 = nn.Linear(l2_units, 1)  # Outputs scalar value V(s)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)  # No activation (linear output)
        return value

class Trainer_ActorCritic:
    def __init__(
            self, env_name, policy_net,
            value_net, config_file, results_path="results"
    ):
        self.env_name = env_name
        self.config = config_file
        self.results_path = results_path
        temp_env = gym.make(env_name)
        self.state_dim = temp_env.observation_space.shape[0]
        self.action_dim = temp_env.action_space.n

        self.policy = policy_net(
            self.state_dim, self.action_dim,
            config_file["model"]["l1_units"],
            config_file["model"]["l2_units"]
        )

        self.value_net = value_net(
            self.state_dim,
            config_file["model"]["l1_units"],
            config_file["model"]["l2_units"]
        )

    def train(self, l_rate=None, max_steps=None, gamma=None, avg_window=None):
        gamma = gamma if gamma is not None else self.config["training"]["gamma"]
        l_rate = l_rate if l_rate is not None else self.config["training"]["lr"]
        max_steps = max_steps if max_steps is not None else self.config["training"]["steps"]
        avg_window = avg_window if avg_window is not None else self.config["training"]["avg_window"]

        env = gym.make(self.env_name)
        optimizer_policy = optim.Adam(self.policy.parameters(), lr=l_rate)
        optimizer_value = optim.Adam(self.value_net.parameters(), lr=l_rate)

        all_rewards, avg_rewards, steps_list = [], [], []
        total_steps = 0
        total_episodes = 0

        pbar = tqdm(total=max_steps)
        while total_steps < max_steps:
            state = env.reset()[0]
            done = False
            log_probs, values, rewards = [], [], []
            steps = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits = self.policy(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                value = self.value_net(state_tensor)

                next_state, reward, done, _, _ = env.step(action.item())

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                state = next_state
                steps += 1
                total_steps += 1

            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns).float().unsqueeze(1)

            values = torch.cat(values)
            log_probs = torch.stack(log_probs)
            advantages = returns - values.detach()

            # Policy loss (Actor)
            policy_loss = -(log_probs * advantages).mean()

            # Value loss (Critic)
            value_loss = F.mse_loss(values, returns)

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            episode_reward = sum(rewards)
            all_rewards.append(episode_reward)
            steps_list.append(steps)
            avg_rewards.append(sum(all_rewards[-avg_window:]) / len(all_rewards[-avg_window:]))

            pbar.set_description(f"Reward: {episode_reward:.1f}, Steps: {steps}")
            pbar.update(steps)

            total_episodes += 1

        pbar.close()
        return avg_rewards, steps_list, total_episodes

if __name__ == "__main__":
    results_path = "results"
    env_name = "CartPole-v1"
    config_path = "config.json"
    config_file = load_config(config_path)

    # Init trainer and run training
    trainer = Trainer_ActorCritic(
        env_name=env_name,
        policy_net=PolicyNet,
        value_net=ValueNet,
        config_file=config_file,
        results_path=results_path
    )
    avg_rewards, steps_list, total_episodes = trainer.train()

    plot_metrics(
        avg_rewards, steps_list, save_path=results_path, figure_name="actor_critic_single_run"
    )
