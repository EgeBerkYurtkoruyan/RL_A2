# Deep Reinforcement Learning Assignment 2

This repository contains the source code for the Deep Reinforcement Learning (DRL) Assignment 2. Before running the experiments, please set up the folder structure as described below. Since files cannot be uploaded in a .zip format, you will need to create the necessary folders manually.


## Setup and Installation
Clone the repository and navigate to its directory.
Install dependencies using pip:

**pip install -r requirements.txt**

## Folder Structure Setup

### 1. `models` Folder
- **Create a folder named `models`.**
- Place the file `model.py` in this folder.
- **Note:** `model.py` contains the `PolicyNet` and `ValueNet` classes that are used in the project.

### 2. `trains` Folder
- **Create a folder named `trains`.**
- Place the following training scripts in this folder:
  - `train_reinforce.py`
  - `train_ac.py`
  - `train_a2c.py`
- **Note:** Each file contains the training code for its respective model.

### 3. `utils` Folder
- **Create a folder named `utils`.**
- This folder should include:
  - A configuration loader module (for example, a script to load configurations using `load_file.py`).
  - Code for visualization and other utility functions required by the project (**plot_mean_results.py** and **visualize.py**).

## Running the Experiment

- The experiment is started by running the following command in your terminal: **python experiment.py**

  
- **Configuration:**  
The script reads hyperparameters and other settings (such as the number of iterations) from `config.json`.
- **Output:**  
Upon execution, the experiment results will be written to an automatically created folder. The naming convention of this folder reflects the current method used for the source file.

## Additional Notes

- **Preparation:** Ensure your directory structure matches the above instructions before running the experiments.
- **Modifications:** Adjust hyperparameters or other settings as needed in the `config.json` file.
