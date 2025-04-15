import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# For each experiment performed in the ablation study calculates the mean of the five iterations, apply the process to all the values tried for
# a given parameter
def plot_experiment_means(folder_path, param_name, save_path=None, figure_name="experiment_means",
                         num_points=200, smoothing_window=10):
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} experiment files")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    
    experiment_means = []
    global_min_step = float('inf')
    global_max_step = float('-inf')
    
    # First pass: determine global min and max steps
    for json_file in json_files:
        with open(json_file, "r") as f:
            ablation_data = json.load(f)
        
        try:
            rewards_reps = ablation_data["rewards"]
            steps_reps = ablation_data["steps"]
        except KeyError as e:
            print(f"Error: Missing key {e} in {json_file}")
            continue
        
        # Find min and max steps across all iterations in this experiment
        for steps in steps_reps:
            if len(steps) > 0:
                global_min_step = min(global_min_step, min(steps))
                global_max_step = max(global_max_step, max(steps))
    
    common_steps = np.linspace(global_min_step, global_max_step, num_points)
    
    # Second pass: process each experiment
    for i, json_file in enumerate(json_files):
        filename = os.path.basename(json_file)
        
        # Extract parameter value from filename using regex
        # Pattern looks for value between "dqn_" and "_data"
        match = re.search(r'_([\d.]+)_data\.json$', filename)
        #match = re.search(r'dqn_(\d+_\d+)', filename)
        if match:
            param_value = match.group(1)
            exp_label = f"{param_name} = {param_value}"
        else:
            # Fallback if pattern doesn't match
            exp_label = os.path.basename(json_file).replace('.json', '')
        
        with open(json_file, "r") as f:
            ablation_data = json.load(f)
        
        rewards_reps = ablation_data["rewards"]
        steps_reps = ablation_data["steps"]
        episodes_reps = ablation_data.get("episodes", [])
        
        # Process each iteration in this experiment
        smoothed_rewards_list = []
        
        for rewards, steps in zip(rewards_reps, steps_reps):
            if len(rewards) >= smoothing_window:
                # Apply smoothing
                smoothed_rewards = np.convolve(rewards, np.ones(smoothing_window) / smoothing_window, mode="valid")
                valid_steps = steps[:len(smoothed_rewards)]
                
                # Interpolate to common step scale
                if len(valid_steps) > 0:
                    interp_rewards = np.interp(common_steps, valid_steps, smoothed_rewards)
                    smoothed_rewards_list.append(interp_rewards)
        
        if smoothed_rewards_list:
            # Calculate mean across all iterations for this experiment
            exp_mean = np.mean(np.array(smoothed_rewards_list), axis=0)
            experiment_means.append((param_value, exp_mean))
            
            # Plot the mean for this experiment
            sns.lineplot(x=common_steps, y=exp_mean, label=exp_label, linewidth=2)
    
    # Labels and title
    plt.xlabel("Total Steps")
    plt.ylabel("Average Reward")
    plt.title(rf"$\bf{{Average\ Rewards\ vs\ Steps\ -\ Ablation\ Study}}$" 
          f"\nRewards Across Different {param_name.upper()} Values", fontsize=14)
    plt.legend(loc="best", frameon=True)
    
    # Save the figure if a path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file_path = os.path.join(save_path, f"{figure_name}.png")
        plt.gcf().set_size_inches(13, 6) 
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")
    
    plt.show()
    
    return experiment_means



def plot_final(folder_path, save_path=None, figure_name="experiment_means",
                         num_points=200, smoothing_window=10):
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} experiment files")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    
    experiment_means = []
    global_min_step = float('inf')
    global_max_step = float('-inf')
    
    # First pass: determine global min and max steps
    for json_file in json_files:
        with open(json_file, "r") as f:
            ablation_data = json.load(f)
        
        try:
            rewards_reps = ablation_data["rewards"]
            steps_reps = ablation_data["steps"]
        except KeyError as e:
            print(f"Error: Missing key {e} in {json_file}")
            continue
        
        # Find min and max steps across all iterations in this experiment
        for steps in steps_reps:
            if len(steps) > 0:
                global_min_step = min(global_min_step, min(steps))
                global_max_step = max(global_max_step, max(steps))
    
    common_steps = np.linspace(global_min_step, global_max_step, num_points)
    
    # Second pass: process each experiment
    for i, json_file in enumerate(json_files):
        filename = os.path.basename(json_file)
        
        # Extract label from filename (before the first underscore)
        match = re.match(r'^(.*?)_experiment', filename)
        exp_label = match.group(1) if match else filename.replace('.json', '')
        exp_label = exp_label.replace('_', '+')

        
        with open(json_file, "r") as f:
            ablation_data = json.load(f)
        
        rewards_reps = ablation_data["rewards"]
        steps_reps = ablation_data["steps"]
        
        # Process each iteration in this experiment
        smoothed_rewards_list = []
        
        for rewards, steps in zip(rewards_reps, steps_reps):
            if len(rewards) >= smoothing_window:
                # Apply smoothing
                smoothed_rewards = np.convolve(rewards, np.ones(smoothing_window) / smoothing_window, mode="valid")
                valid_steps = steps[:len(smoothed_rewards)]
                
                # Interpolate to common step scale
                if len(valid_steps) > 0:
                    interp_rewards = np.interp(common_steps, valid_steps, smoothed_rewards)
                    smoothed_rewards_list.append(interp_rewards)
                    
                    # Plot individual repetitions
                    #sns.lineplot(x=common_steps, y=interp_rewards, alpha=0.3, linewidth=1, color='gray')
        
        if smoothed_rewards_list:
            # Calculate mean across all iterations for this experiment
            exp_mean = np.mean(np.array(smoothed_rewards_list), axis=0)
            experiment_means.append((exp_label, exp_mean))
            
            # Plot the mean for this experiment
            sns.lineplot(x=common_steps, y=exp_mean, label=exp_label, linewidth=2)
    
    # Labels and title
    plt.xlabel("Total Steps")
    plt.ylabel("Average Reward")
    plt.title(r'$\bf{Average\ Reward\ vs\ Steps:\ Methods\ Implementation}$', fontsize=14)  # Bold title
    plt.legend(loc="best", frameon=True)
    
    # Save the figure if a path is provided
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file_path = os.path.join(save_path, f"{figure_name}.png")
        plt.gcf().set_size_inches(13, 6) 
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")
    
    plt.show()
    
    return experiment_means






if __name__ == "__main__":
    param_name = "lr"
    fig_name = "ablation"+"_"+param_name
    results_path = "results/REINFORCE/ablation"
    folder_path = os.path.join(results_path,param_name,"data")
    save_path = "results/paper_results"

    plot_experiment_means(folder_path=folder_path, param_name = param_name,save_path=save_path, figure_name=fig_name,
                         num_points=200, smoothing_window=50)

    # To plot the mean of the different training methodogies run function plot_final
    # Add the path where you saved your reuslts up to the data folder resulst/name_of_folder/experiment/data
    #folder_path = "results/final/experiment/data"
    #save_path = "results/paper_results"
    #plot_final(folder_path, save_path=save_path, figure_name="DQN_ER_TN_test",
    #                     num_points=200, smoothing_window=100)

