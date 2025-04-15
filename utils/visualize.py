import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

def plot_metrics(rewards_list, steps_list, episodes_list, 
                 save_path=None, figure_name=None, 
                 labels=None, num_points=200, 
                 smoothing_window=50,
                 model=None):

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    num_iterations = len(rewards_list)
    if labels is None:
        labels = [f"Run {i+1}" for i in range(num_iterations)]

    # Smooth rewards
    smoothed_rewards_list = [
        np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode="valid")
        for rewards in rewards_list
    ]

    # Common interpolation steps
    min_step = min(min(steps) for steps in steps_list)
    max_step = max(max(steps) for steps in steps_list)
    common_steps = np.linspace(min_step, max_step, num_points)

    interpolated_rewards = []
    for i in range(num_iterations):
        interp_rewards = np.interp(common_steps, steps_list[i][:len(smoothed_rewards_list[i])], smoothed_rewards_list[i])
        interpolated_rewards.append(interp_rewards)
        sns.lineplot(x=common_steps, y=interp_rewards, label=labels[i], alpha=0.5)

    interpolated_rewards = np.array(interpolated_rewards)

    # Stats
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_rewards = np.std(interpolated_rewards, axis=0)
    avg_reward = np.mean(mean_rewards)
    avg_episodes = np.mean(episodes_list)
    last_reward = mean_rewards[-1]

    print(f"Average reward: {avg_reward}")
    print(f"Average number of episodes: {avg_episodes}")
    print(f"Last obtained reward: {last_reward}")

    # Plot mean line
    sns.lineplot(x=common_steps, y=mean_rewards, label="Mean Reward", color="black", linewidth=2)

    # Shaded std area
    plt.fill_between(common_steps, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     color="black", alpha=0.1, label="Std Dev")

    # Labels and title
    plt.xlabel("Total Steps")
    plt.ylabel("Average Reward")
    title = "Training Progress: Average Reward vs. Steps"
    if model:
        title += f" ($\\mathbf{{Layer\\ units}}$ : {' | '.join(map(str, model))})"
    plt.title(title)

    plt.legend(loc="best", frameon=True)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, figure_name + ".png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")

    plt.show()
    
    return avg_reward, avg_episodes, last_reward


if __name__ == "__main__":
    folder_path = "results/A2C/experiment/data"
    file_name = "a2c_exp_data.json"
    file_path = os.path.join(folder_path, file_name)

    save_path = "results/REINFORCE/experiment/plot"

    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract data
    rewards_list = data["rewards"]
    steps_list = data["steps"]
    episodes_list = data["episodes"]

    plot_metrics(rewards_list, steps_list, episodes_list, 
                 save_path=save_path, 
                 figure_name="ac_run_plot")
