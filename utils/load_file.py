import json

def load_config(file_path):
    try:
        with open(file_path) as f:
            d = json.load(f)
            return d
    except Exception as e:
        print(f"Error loading config file: {e}")
