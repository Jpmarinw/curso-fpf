import pickle
import yaml

def load_config():
    with open("config/prod.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)