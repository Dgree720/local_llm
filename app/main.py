import requests
from utils.download_models import download_models
from utils.configuration import config


config = config()
required_models = [config["required_models"][i] for i in config["required_models"]]
models_path = config["model_path"]

download_models(models_path, required_models)
