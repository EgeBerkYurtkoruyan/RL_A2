## Deep learning assignment 1

### Andres Aranguren s4403290


## Set up environment

## 1. Set up python environment


## 1. Install Python
Ensure you have Python installed. You can check by running:
```sh
python --version
```
or
```sh
python3 --version
```

If Python is not installed, download and install it from [python.org](https://www.python.org/downloads/).

## 2. Create a Virtual Environment
Navigate to your project directory and create a virtual environment:
```sh
python -m venv myenv
```
or (for Python 3)
```sh
python3 -m venv myenv
```
## 3. Activate the Virtual Environment
### On Windows:
```sh
myenv\\Scripts\\activate
```
### On macOS and Linux:
```sh
source myenv/bin/activate
```


## 4. Install Dependencies
Once the environment is activated, install dependencies as needed, e.g.:
```sh
pip install -r requirements.txt
```


## Run experiments

To run experiments run the follwing python file

```sh
experiment.py
```

## 1. Set exp name and saving results path


```sh
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
```

- The results of the experiment will be saved in the results folder inside a subfolder named with the exp name. Such as results/test

## 2. Set the experiment




- For the expeirment define the Training meethod you want to implement. Possible values

    1. Trainer_Naive
    2. Trainer_buffer
    3. Trainer_target_network
    4. Trainer_buffer_target
    5. Trainer_ddqn

```sh
exp = Experiment(method=Trainer_Naive,env_name = env_name,exp_name = exp_name ,config_path = config_path,results_path=results_path)

```
## 3. Run experiment


- When running this line will run 5 repetitions and will save the list with the values as JSON file in the specified path

- Results will be stored in the a folder inside resuls/exp_name/experiment/data/file.json 


```sh

exp.run_experiment(save_results=True)

```

## 4. Run ablation
- When running ablation specify the parameter for whihc you wnat to perform an abltion param_name

- Specify the values you want to test in a list eg [0.1,0.01,0.001]

- If you are using one of the methodologies like ER, TN, both, or DDQN set methods to True

- Results will be stored in the a folder inside resuls/exp_name/experiment/ablation/param_name/data

```sh
test = exp.run_ablation(section="training",param_name = "lr",param_values = [0.1],save_results=True, methods=False)
```

## 5. Plot results

Once you run experiments for different training  methods  see .section 3

Go to:

```sh
plot_mean_results.py
```
Then run the file setting up like:

- To plot the mean of the different training methodogies run function plot_final
- Add the path where you saved your reuslts up to the data folder resulst/name_of_folder/experiment/data

```sh

folder_path = "results/test/experiment/data",save_path = "results/paper_results"

plot_final(folder_path, save_path=save_path, figure_name="DQN_ER_TN_test",num_points=200, smoothing_window=100)
```



    


