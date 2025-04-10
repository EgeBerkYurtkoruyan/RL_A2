from train import Trainer_Naive
from train_buffer import Trainer_buffer
from train_target_network import Trainer_target_network
from train_buffer_target import Trainer_buffer_target
from train_double_dqn import Trainer_ddqn
from utils.load_file import load_config, save_metrics
from models.dqmodel import Qnet
from utils.visualize import plot_metrics
import os
import copy

class Experiment:

    def __init__(self,env_name:str=None,exp_name:str=None,config_path:str=None,results_path:str=None,method=None):
        self.exp_name = exp_name
        self.results_path = results_path
        self.env_name = env_name
        self.config_path = config_path
        self.config_file = load_config(self.config_path)
        self.iterations = self.config_file["training"]["iterations"]
        self.max_steps = self.config_file["training"]["steps"]
        self.method=method
        self.exp_path = os.path.join(results_path,exp_name)
    def run_experiment(self, save_results: bool = False):
        print("Running single experiment with default configuration settings.")

        # Define paths for saving results
        experiment_path = os.path.join(self.results_path, "experiment")
        os.makedirs(experiment_path, exist_ok=True)

        data_path = os.path.join(experiment_path, "data")
        plot_path = os.path.join(experiment_path, "plot")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        # Load configuration
        custom_config = copy.deepcopy(self.config_file)
        trainer = self.method(self.env_name, Net=Qnet,
                        config_file=custom_config,
                        results_path=self.results_path)
        rewards_reps, steps_reps, episodes_reps = trainer.train_repetitions(num_iterations=self.iterations)

        # Structure results
        experiment_data = {
                "rewards": rewards_reps,
                "steps": steps_reps,
                "episodes": episodes_reps
        }

        # Generate a filename based on experiment name
        file_name = self.exp_name + "_experiment_data"

        if save_results:
            # Save metrics
            save_metrics(data=experiment_data, results_path=data_path, filename=file_name)

        return experiment_data

    def run_ablation(self,section=None,param_name=None,param_values=None,save_results:bool=False,
                     model_config:bool=False,class_method=None,
                     methods:bool=False):
        print(f"Running ablation for parameter {param_name}")
        ablation_path = os.path.join(self.results_path,"ablation")
        if not os.path.exists(ablation_path):
            os.mkdir(ablation_path)
        if section=="model":
            param_path = os.path.join(ablation_path,"layer_units")
        else:
            param_path = os.path.join(ablation_path,param_name)
        if not os.path.exists(param_path):
            os.mkdir(param_path) 
        
        data_path = os.path.join(param_path,"data")
        plot_path = os.path.join(param_path,"plot")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
          


        ablation_results = {}

        for it,value in enumerate(param_values):
            print(f"Training value: {value}")
            custom_config = copy.deepcopy(self.config_file)
            if section == "model":
                l1_name,l2_name = param_name
                l1_value,l2_value = value[0],value[1]
                custom_config[section][l1_name] = l1_value
                custom_config[section][l2_name] = l2_value
            else:
                custom_config[section][param_name] = value

            # Initialize Trainer
            trainer = self.method(self.env_name,Net = Qnet,
                          config_file=custom_config,
                          results_path=self.results_path)
            rewards_reps,steps_reps,episodes_reps = trainer.train_repetitions(num_iterations=self.iterations)
            ablation_reps = (rewards_reps,steps_reps,episodes_reps)
            ablation_data = {
            "rewards": rewards_reps,
            "steps": steps_reps,
            "episodes": episodes_reps
            }
            if section == "model":
                value_dict = "net_"+str(value[0])+"_"+str(value[1])
                ablation_results[value_dict] = ablation_data
                file_name = self.exp_name + "_" + "_".join(map(str, value))

            else:
                ablation_results[value] = ablation_data
                file_name = self.exp_name+"_"+str(value)+"_data"
            

            if save_results:
                # Save metrics
                save_metrics(data=ablation_data,results_path=data_path,filename =file_name )
                #plot_metrics(save_path = plot_path,figure_name = file_name,ablation_reps=ablation_reps, labels=None, num_points=200,smoothing_window=10,param_name = param_name,param_value=value,model=model_config,
                 #            methods=methods)


        return ablation_data 

        
if __name__ == "__main__":

    if not os.path.exists("results"):
        os.mkdir("results")
    # Env name
    env_name = 'CartPole-v1'
    # For final experiments run this line DQN, DQN_naive, DQN_buffer....
    results_path = "results/test"
    # For ablation studies uncomment line under
    #results_path = "results/ablation"
    # Configle file path
    config_path = 'config.json'
    # Set your experiment name like DQN, DQN_ER, DQN_TN, DQN_ER_TN
    exp_name = "test"

    ## 1. Open config.json and set your corresponding hyperparameters for the model architecture and training parameters

    #Initialize experiment in method set the method you want to try choose from the follwoing:
    '''
    1. Trainer_Naive
    2. Trainer_buffer
    3. Trainer_target_network
    4. Trainer_buffer_target
    5. Trainer_ddqn
    '''
    ## Running for Naive
    exp = Experiment(method=Trainer_Naive,env_name = env_name,exp_name = exp_name ,config_path = config_path,results_path=results_path)
    # Run experiment for Naive DQN
    # When running this line will run 5 repetitions and will save the list with the values as JSON file in the specified path
    # Results will be stored in the a folder inside resuls/exp_name/experiment/data/file.json 
    exp.run_experiment(save_results=True)
    #1. Repeat process for the other training methods
    ## When running ablation specify the parameter for whihc you wnat to perform an abltion param_name
    ## Specify the values you want to test in a list eg [0.1,0.01,0.001]
    ## If you are using one of the methodologies like ER, TN, both, or DDQN set methods to True

    ## Results will be stored in the a folder inside resuls/exp_name/experiment/ablation/param_name/data
    test = exp.run_ablation(section="training",param_name = "lr",
                                    param_values = [0.1],
                                    save_results=True,
                                    methods=False)
    
    ## Possible values to test
    #lr = [0.01,0.001,0.003,0.0005,0.0001]
    # epsilon = [1.0, 0.75, 0.5, 0.25, 0.1]
    # n_size = [[4,8],[8,16],[16,32],[32,64],[64,128]]
    #update_ratio = [1,4,8,16,32]
    
    #er_tn_data = exp.run_ablation(section="training",param_name = "avg_window",
    #                                param_values = [100],
    #                                save_results=True,
     #                               methods=True)
    #tn_data = exp.run_ablation(section="training",param_name = "target_network_update",
    #                                param_values = [200,500,1000,2000],
    #                                save_results=True,
    #                                methods=True)

    #exp_data= exp.run_experiment(save_results=True,plot_results=True)

    ## Run the ablation study

    # Ablation for the learning rate
    #l_rate_data = exp.run_ablation(section="training",param_name = "lr",
    #                                 param_values = [0.0001],
    #                                 save_results=True)
    # Ablation for the exploration factor
    #exploration_data = exp.run_ablation(section="training",param_name = "epsilon",
    #                                param_values = [0.1,0.25,0.5,0.75,1.0],
    #                                 save_results=True)
    #ratio_data = exp.run_ablation(section="training",param_name = "update_ratio",
    #                                param_values = [16],
    #                                 save_results=False)
    #ratio_data = exp.run_ablation(section="training",param_name = "epsilon_decay",
    #                               param_values = [0.5],
    #                                 save_results=True)
    
    #buffer_data = exp.run_ablation(section="training",param_name = "replay_N",
    #                                param_values = [64],
    #                                 save_results=True)
    # Ablation for the network size
    #network_data = exp.run_ablation(section="model",param_name = ["l1_units","l2_units"],
    #                               param_values = [[64,128]],
    #                                save_results=True, model_config=True)
    #target_net = exp.run_ablation(section="training",param_name = "target_network_update",
    #                               param_values = [100],
     #                                save_results=True)
    
    
    #lr = [0.01,0.001,0.003,0.0005,0.0001]
    # epsilon = [1.0, 0.75, 0.5, 0.25, 0.1]
    # n_size = [[4,8],[8,16],[16,32],[32,64],[64,128]]
    #update_ratio = [1,4,8,16,32]
    
    


