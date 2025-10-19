import os
import yaml
from pathlib import Path


def config():
    try:
        parent = Path.cwd().parent
        config_path = f"{parent}/config.yaml"
        models_path = f"{parent}/models"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["model_path"] = models_path

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        print(f"Error occured:\n{e}")
