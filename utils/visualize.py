import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(rewards, steps, 
                            save_path=None, 
                            figure_name="reinforce_single", 
                            smoothing_window=10, 
                            num_points=200):
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    # Smooth rewards using moving average
    smoothed_rewards = np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode="valid")
    steps = steps[:len(smoothed_rewards)]  # Match lengths

    # Interpolate to fixed number of points
    interp_steps = np.linspace(min(steps), max(steps), num_points)
    interp_rewards = np.interp(interp_steps, steps, smoothed_rewards)

    # Plot
    sns.lineplot(x=interp_steps, y=interp_rewards, label="Smoothed Reward", color="blue", linewidth=2)
    
    plt.xlabel("Total Steps")
    plt.ylabel("Average Episode Reward")
    plt.title("Training Progress: REINFORCE (Single Run)")
    plt.grid(True)
    plt.legend(loc="best")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, f"{figure_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
    
    plt.show()

    print(f"Final reward: {rewards[-1]}")
    print(f"Mean of last 10 episodes: {np.mean(rewards[-10:])}")
