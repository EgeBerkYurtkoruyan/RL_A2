import json
import os

def load_config():
    try:
        return{
            "model": {
                "l1_units": 128,
                "l2_units": 256
            },
            "training": {
                "iterations":5,
                "lr": 0.0001,
                "gamma": 0.99,
                "steps": 1000000,
                "episodes": 10000,
                "avg_window": 100,
                "n_steps":1,
                "advantage": True
            }
        }
    except Exception as e:
        print(f"Error loading config file: {e}")


def save_metrics(data, results_path: str = None,filename :str=None):
    file_path = os.path.join(results_path,filename+".json")
    with open(file_path,"w") as f:
        json.dump(data,f)
    print("Saved data in:", file_path)