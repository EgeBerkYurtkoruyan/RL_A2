from utils.load_file import load_config, save_metrics
import os
from train.train_reinforce import Trainer_Reinforce
from train.train_ac import Trainer_ActorCritic
from train.train_a2c import Trainer_AdvantageActorCritic
from models.model import PolicyNet, ValueNet
from dotenv import load_dotenv
import copy  

class Experiment:
    def __init__(self,
                 method:str =  None, 
                 env_name:str=None,
                 exp_name:str=None,
                 config_path:str=None):
        self.method = method
        self.exp_name = exp_name
        self.env_name = env_name
        self.config_path = config_path
        self.config_file = load_config(self.config_path)
        self.iterations = self.config_file["training"]["iterations"]
        self.max_steps = self.config_file["training"]["steps"]
        self.advantage = self.config_file["training"]["advantage"]

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
                                            config_file=self.config_file,
                                            advantage=self.advantage)
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
    


    def run_ablation(self, section=None, param_name=None, param_values=None,
                 save_results: bool = False, model_config: bool = False,
                 class_method=None, methods: bool = False):
    
        print(f"Running ablation for parameter {param_name}")
        
        # Define base path: results/<METHOD>/ablation/
        ablation_path = os.path.join("results", self.method, "ablation")
        os.makedirs(ablation_path, exist_ok=True)

        if section == "model":
            param_path = os.path.join(ablation_path, "layer_units")
        else:
            param_path = os.path.join(ablation_path, param_name)
        os.makedirs(param_path, exist_ok=True)

        data_path = os.path.join(param_path, "data")
        plot_path = os.path.join(param_path, "plot")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        ablation_results = {}

        for it, value in enumerate(param_values):
            print(f"Training value: {value}")
            custom_config = copy.deepcopy(self.config_file)

            # Apply config change
            if section == "model":
                l1_name, l2_name = param_name
                l1_value, l2_value = value[0], value[1]
                custom_config[section][l1_name] = l1_value
                custom_config[section][l2_name] = l2_value
            else:
                custom_config[section][param_name] = value

            # Initialize trainer based on method string
            if self.method == "REINFORCE":
                trainer = Trainer_Reinforce(
                    env_name=self.env_name,
                    Net=PolicyNet,
                    config_file=custom_config
                )
            elif self.method == "AC":
                trainer = Trainer_ActorCritic(
                    env_name=self.env_name,
                    PolicyClass=PolicyNet,
                    ValueClass=ValueNet,
                    config_file=custom_config
                )
            elif self.method == "A2C":
                trainer = Trainer_AdvantageActorCritic(
                    env_name=self.env_name,
                    PolicyClass=PolicyNet,
                    ValueClass=ValueNet,
                    config_file=custom_config,
                    advantage=self.advantage
                )
            else:
                raise ValueError(f"Unsupported method: {self.method}")

            # Train
            rewards_reps, steps_reps, episodes_reps = trainer.train_repetitions(num_iterations=self.iterations)
            ablation_reps = (rewards_reps, steps_reps, episodes_reps)

            ablation_data = {
                "rewards": rewards_reps,
                "steps": steps_reps,
                "episodes": episodes_reps
            }

            if section == "model":
                value_dict = f"net_{value[0]}_{value[1]}"
                ablation_results[value_dict] = ablation_data
                file_name = self.exp_name + "_" + "_".join(map(str, value))
            else:
                ablation_results[str(value)] = ablation_data
                file_name = self.exp_name + "_" + str(value) + "_data"

            if save_results:
                save_metrics(data=ablation_data, results_path=data_path, filename=file_name)
                # Optionally plot_metrics()

        return ablation_results


    


if __name__ == "__main__":
    load_dotenv()

    if not os.path.exists("results"):
        os.mkdir("results")
    env_name = os.getenv("ENV_NAME") # Environment name
    config_path = os.getenv("CONFIG_PATH") # Path to config file

    method = "REINFORCE" # Define the method between REINFORCE , AC, A2C

    exp_name = "reinforce_1" # Name of experiment

    # Initialize experiment
    exp = Experiment(method=method,
    env_name = env_name,
                     exp_name = exp_name ,
                     config_path = config_path)
    # Run experiment
    #exp.run_experiment(save_results=True)
    # Define the values for the ablation study
    lr_vals = [0.01,0.001,0.003,0.0005,0.0001]
    l_rate_data = exp.run_ablation(section="training",param_name = "lr",
                                     param_values = lr_vals,
                                     save_results=True)





