import json
import os

def load_config(file_path):
    try:
        with open(file_path) as f:
            d = json.load(f)
            return d
    except Exception as e:
        print(f"Error loading config file: {e}")


def save_metrics(data, results_path: str = None,filename :str=None):
    file_path = os.path.join(results_path,filename+".json")
    with open(file_path,"w") as f:
        json.dump(data,f)
    print("Saved data in:", file_path)