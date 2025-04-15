# RL Master Course Assignment: Policy Gradient and Actor-Critic Methods

This project implements several reinforcement learning algorithms for the CartPole environment including the REINFORCE algorithm, basic Actor-Critic (AC), and Advantage Actor-Critic (A2C). 

## Project Structure

- **Training Scripts**  
  - `train_reinforce.py`: Implements the REINFORCE algorithm.  
  - `train_ac.py`: Implements the basic Actor-Critic (AC) algorithm.  
  - `train_a2c.py`: Implements the Advantage Actor-Critic (A2C) algorithm with advantage calculations.

- **Model and Utility Files**  
  - `model.py`: Contains the definitions for `PolicyNet` and `ValueNet`, the neural networks used for the policy and value approximations. 
  - `load_file.py`: Provides utility functions to load the configuration file and save metrics. 
  - `visualize.py`: Offers functions to plot training metrics such as rewards over time. 
  - `plot_mean_results.py`: Contains functions to visualize the mean results from multiple experiment repetitions. 

- **Configuration and Dependencies**  
  - `requirements.txt`: Lists all Python dependencies required to run the project.

- **Experiment Management**  
  - `experiment.py`: A script to run experiments with multiple repetitions and ablation studies, allowing an analysis of performance across different hyperparameters.
  - You can run the the experiment code by
    ```bash
    python experiment.py

## Requirements

Your folder structure should look like 

```
📁 RL_A2/
├── 📁 models/                 # Contains model definitions
│   └── model.py
├── 📁 results/                # Directory for storing results
├── 📁 results_final/          # Directory for storing final results
├── 📁 train/                  # Training scripts for different algorithms
│   ├── train_a2c.py
│   ├── train_ac.py
│   └── train_reinforce.py
├── 📁 utils/                  # Utility functions (e.g., logging, plotting)
│   ├── load_file.py
│   ├── plot_mean_results.py
│   └── visualize.py
├── experiment.py             # Entry point or experiment runner
├── view_results.py           # Script to visualize or analyze results
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

```

Make sure you have Python 3.12 or higher installed. Then, install the dependencies using pip:

```bash
pip install -r requirements.txt
