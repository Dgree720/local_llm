import os
import yaml
from pathlib import Path
import sys


def config():
    try:
        parent = Path.cwd().parent
        config_path = f"{parent}/config.yaml"
        if os.path.isdir(f"{parent}/models"):
            models_path = f"{parent}/models"
        else:
            os.mkdir(f"{parent}/models")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # define paths
        models_path = f"{parent}/models"
        vector_store = os.path.join(os.getcwd(), "chroma_db")

        # write paths to config
        config["model_path"] = models_path

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        print(f"Error occurred:\n{e}")
