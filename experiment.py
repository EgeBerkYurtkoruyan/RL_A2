from utils.load_file import load_config, save_metrics
import os
from train.train_reinforce import Trainer_Reinforce
from train.train_ac import Trainer_ActorCritic
from train.train_a2c import Trainer_AdvantageActorCritic
from models.model import PolicyNet, ValueNet
from dotenv import load_dotenv


class Experiment:
    def __init__(self,
                 method:str =  None, 
                 env_name:str=None,
                 exp_name:str=None,
                 config_path:str=None,
                 PolicyNet=None,
                 ValueNet=None):
        self.method = method
        self.exp_name = exp_name
        self.env_name = env_name
        self.config_path = config_path
        self.config_file = load_config(self.config_path)
        self.iterations = self.config_file["training"]["iterations"]
        self.max_steps = self.config_file["training"]["steps"]

    def run_experiment(self, save_results: bool = False):
        print("Running single experiment with default configuration settings.")
        # Define paths for saving results
        experiment_path = os.path.join(os.path.join("results",self.method), "experiment")
        os.makedirs(experiment_path, exist_ok=True)
        data_path = os.path.join(experiment_path, "data")
        plot_path = os.path.join(experiment_path, "plot")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        ## Define method between REINFORCE and Actor-Critic and A2C
        if self.method == "REINFORCE":
            print("Running REINFORCE method.")
            trainer = Trainer_Reinforce(env_name=self.env_name,
                                         Net = PolicyNet, 
                                         config_file=self.config_file)
        elif self.method == "AC":
            print("Running Actor-Critic method.")
            trainer = Trainer_ActorCritic(env_name = self.env_name,
                                            PolicyClass=PolicyNet, 
                                            ValueClass=ValueNet, 
                                            config_file=self.config_file)
        elif self.method == "A2C":
            print("Running A2C method.")
            trainer = Trainer_AdvantageActorCritic(env_name = self.env_name,
                                            PolicyClass=PolicyNet, 
                                            ValueClass=ValueNet, 
                                            config_file=self.config_file)
        else:
            print("Method not defined. Please use REINFORCE, AC or A2C.")
        

        
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
    load_dotenv()

    if not os.path.exists("results"):
        os.mkdir("results")
    env_name = os.getenv("ENV_NAME") # Environment name
    config_path = os.getenv("CONFIG_PATH") # Path to config file

    method = "A2C" # Define the method between REINFORCE , AC, A2C

    exp_name = "ac_5_it_lr" # Name of experiment

    # Initialize experiment
    exp = Experiment(method=method,
    env_name = env_name,
                     exp_name = exp_name ,
                     config_path = config_path)
    # Run experiment
    exp.run_experiment(save_results=True)


