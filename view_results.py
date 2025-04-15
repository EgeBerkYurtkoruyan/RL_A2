import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_multiple_experiments_mean(json_paths, smoothing_window=30, num_points=200, save_path=None, figure_name="final_comparison"):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    global_min_step = float('inf')
    global_max_step = float('-inf')

    all_data = []

    # First pass: we are determining the global step range
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)
        steps_reps = data["steps"]
        for steps in steps_reps:
            if steps:
                global_min_step = min(global_min_step, min(steps))
                global_max_step = max(global_max_step, max(steps))

    common_steps = np.linspace(global_min_step, global_max_step, num_points)

    # Second pass: processing  each file
    for json_path in json_paths:
        filename = os.path.basename(json_path)
        label = filename.split('_')[0]  # e.g., "reinforce_exp_data.json" → "reinforce"

        with open(json_path, "r") as f:
            data = json.load(f)

        rewards_reps = data["rewards"]
        steps_reps = data["steps"]

        smoothed_rewards_list = []

        for rewards, steps in zip(rewards_reps, steps_reps):
            if len(rewards) >= smoothing_window:
                smoothed_rewards = np.convolve(rewards, np.ones(smoothing_window) / smoothing_window, mode="valid")
                valid_steps = steps[:len(smoothed_rewards)]
                if len(valid_steps) > 0:
                    interp_rewards = np.interp(common_steps, valid_steps, smoothed_rewards)
                    smoothed_rewards_list.append(interp_rewards)

        if not smoothed_rewards_list:
            print(f"No valid runs to plot for {label}.")
            continue

        stacked = np.stack(smoothed_rewards_list)
        mean_rewards = np.mean(stacked, axis=0)
        std_rewards = np.std(stacked, axis=0)

        plt.plot(common_steps, mean_rewards, label=label.upper(), linewidth=2)
        plt.fill_between(common_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

    plt.xlabel("Total Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward ± Std Dev — Final Comparison")
    plt.legend()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{figure_name}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_file}")
    
    plt.show()


json_paths = [
    "results_final/reinforce_exp_data.json",
    "results_final/ac_exp_data.json",
    "results_final/a2c_exp_data.json"
]

plot_multiple_experiments_mean(
    json_paths=json_paths,
    smoothing_window=30,
    save_path="results_final/plots",
    figure_name="final_model_comparison"
)
