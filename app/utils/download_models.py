# Load model directly
from transformers import AutoTokenizer, AutoModel
import os
import yaml


def download_models(model_path, required_models):
    models = set([model for model in os.listdir(model_path)])
    for model in required_models:
        if model:
            if set(model) >= models:
                while True:
                    i = input(
                        f"{model} not downloaded. Do you want do download now? NOTE: Internet connection is required for this (y/n): "
                    ).lower()
                    try:
                        if i in ["y", "yes"]:
                            AutoTokenizer.from_pretrained(
                                "ibm-granite/granite-embedding-107m-multilingual",
                                cache_dir=model_path,
                            )
                            AutoModel.from_pretrained(
                                "ibm-granite/granite-embedding-107m-multilingual",
                                cache_dir=model_path,
                            )
                            print(f"Successfully downloaded {model}")
                            break
                        elif i in ["n", "no"]:
                            print("okay, exiting")
                            break
                    except:
                        print("Wrong input, try again")
                        continue
            else:
                print("All required models available")
