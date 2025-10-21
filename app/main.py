import requests
from utils.download_models import download_models
from utils.configuration import config
from prompts import make_prompt
from llm_service import start_ollama_server, fetch_llm


config = config()
required_models = [config["required_models"][i] for i in config["required_models"]]
models_path = config["model_path"]

download_models(models_path, required_models)
