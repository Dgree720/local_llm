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

        # define configurations to change   
        user_os = sys.platform
        models_path = f"{parent}/models"
        vector_store = os.path.join(os.getcwd(), "chroma_db")
        knowledge_path: /home/andreas/Documents/Python/local_llm/documents
        embed_model: "ibm-granite/granite-embedding-107m-multilingual"
        embedding_model_path: /home/andreas/Documents/Python/local_llm/models/ibm-granite/granite-embedding-107m-multilingual
        vector_db_path: /home/andreas/Documents/Python/local_llm/app/chroma_db
        

        # write paths to config
        config["model_path"] = models_path

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        print(f"Error occurred:\n{e}")
