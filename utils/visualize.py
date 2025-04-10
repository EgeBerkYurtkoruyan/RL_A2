import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(save_path = None,
                 figure_name = None,
                 ablation_reps = None, 
                 labels=None, num_points=200, 
                 smoothing_window=10,
                 param_name=None,
                 param_value=None,model=None, methods=None):

    sns.set_style("whitegrid")  # Clean background
    plt.figure(figsize=(10, 5))
    rewards_list,steps_list,episodes_list = ablation_reps

    num_iterations = len(rewards_list)
    if labels is None:
        labels = [f"Iteration {i+1}" for i in range(num_iterations)]

    # Apply rolling window smoothing BEFORE interpolation
    smoothed_rewards_list = [
        np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode="valid")
        for rewards in rewards_list
    ]

    # Define a common step range from min to max steps
    min_step = min(min(steps) for steps in steps_list)
    max_step = max(max(steps) for steps in steps_list)
    common_steps = np.linspace(min_step, max_step, num_points)

    # Interpolate each run to the common step values
    interpolated_rewards = []
    for i in range(num_iterations):
        interp_rewards = np.interp(common_steps, steps_list[i][:len(smoothed_rewards_list[i])], smoothed_rewards_list[i])
        interpolated_rewards.append(interp_rewards)
        sns.lineplot(x=common_steps, y=interp_rewards, label=labels[i], alpha=0.5)

    # Convert to NumPy array for statistical calculations
    interpolated_rewards = np.array(interpolated_rewards)

    # Compute statistics
    mean_rewards = np.mean(interpolated_rewards, axis=0)  # Compute mean across aligned iterations
    std_rewards = np.std(interpolated_rewards, axis=0)    # Compute standard deviation across runs
    avg_reward = np.mean(mean_rewards)  # Overall mean reward
    avg_episodes = np.mean(episodes_list)  # Average number of episodes
    last_reward = mean_rewards[-1]  # Last mean reward value
    print(f"Average reward: {avg_reward}")
    print(f"Average number of episodes: {avg_episodes}")
    print(f"Last obtained reward: {last_reward}")

    # Plot mean reward curve (smoothed)
    sns.lineplot(x=common_steps, y=mean_rewards, label="Mean Reward", color="black", linewidth=2)

    # Plot standard deviation as a shaded area around the mean (smoothed)
    plt.fill_between(common_steps, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     color="black", alpha=0.1, label="Std Dev")

    # Labels and title
    plt.xlabel("Total Steps")
    plt.ylabel("Average Reward")
    # Add parameter name and value to the title

    title = "Training Progress: Average Reward vs. Steps"
    if methods:
        #title += " " + r"$\mathbf{DQN\ Function\ approximation}$"
        title += " "+f"$\\mathbf{{Target\\ Network\\ Update\\ N = {param_value}}}$"
    if model:
        title += f" ($\\mathbf{{Layer\ units}}$ : {' | '.join(map(str, param_value))})"
    else:
        if param_name is not None and param_value is not None:
            temp =param_name.split('_')  # Replaces underscores with a single space
            title += f" ($\\mathbf{{{temp[0]+" "+temp[1]}}}$ = $\\mathbf{{{param_value}}}$)"     
    plt.title(title)
    # Customizing legend
    plt.legend(loc="best", frameon=True)
    # Save the figure if a path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig_name = figure_name+".png"
        file_path = os.path.join(save_path,fig_name)
        # Save the figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    
    return avg_reward, avg_episodes, last_reward


