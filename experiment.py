from utils.load_file import load_config, save_metrics
import os
from train_reinforce import Trainer_Reinforce
from models.model import PolicyNet

class Experiment:
    def __init__(self, 
                 env_name:str=None,
                 exp_name:str=None,
                 config_path:str=None,
                 results_path:str=None,
                 Net=None):
        self.exp_name = exp_name
        self.results_path = results_path
        self.env_name = env_name
        self.config_path = config_path
        self.config_file = load_config(self.config_path)
        self.iterations = self.config_file["training"]["iterations"]
        self.max_steps = self.config_file["training"]["steps"]
        self.Net = Net

    def run_experiment(self, save_results: bool = False):
        print("Running single experiment with default configuration settings.")
        # Define paths for saving results
        experiment_path = os.path.join(self.results_path, "experiment")
        os.makedirs(experiment_path, exist_ok=True)
        data_path = os.path.join(experiment_path, "data")
        plot_path = os.path.join(experiment_path, "plot")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        trainer = Trainer_Reinforce(self.env_name,self.Net,self.config_file)
        # Structure results
        rewards_reps, steps_reps, episodes_reps = trainer.train_repetitions(num_iterations=self.iterations)

        experiment_data = {
                "rewards": rewards_reps,
                "steps": steps_reps,
                "episodes": episodes_reps
        }
        file_name = self.exp_name + "_data"
        if save_results:
            # Save metrics
            save_metrics(data=experiment_data, results_path=data_path, filename=file_name)
        return experiment_data
    



if __name__ == "__main__":

    if not os.path.exists("results"):
        os.mkdir("results")
    # Env name
    env_name = 'CartPole-v1'
    # For final experiments run this line DQN, DQN_naive, DQN_buffer....
    results_path = "results/reinforce"
    # For ablation studies uncomment line under
    #results_path = "results/ablation"
    # Configle file path
    config_path = 'config.json'
    # Set your experiment name like DQN, DQN_ER, DQN_TN, DQN_ER_TN
    exp_name = "test"
    exp = Experiment(Net = PolicyNet,env_name = env_name,exp_name = exp_name ,config_path = config_path,results_path=results_path)
    exp.run_experiment(save_results=True)


