# Load model directly
from transformers import AutoTokenizer, AutoModel
import os
from huggingface_hub import snapshot_download


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
                            try:
                                snapshot_download(
                                    repo_id=model,
                                    local_dir=f"{model_path}/{model}",
                                    local_dir_use_symlinks=False,
                                    revision="main",  # optional, can be a specific tag/commit
                                    ignore_patterns=[
                                        "*.lock"
                                    ],  # avoids cache artifacts
                                )

                                print(f"Successfully downloaded {model}")
                                break
                            except Exception as e:
                                print(f"Error occurred:\n{e}")
                        elif i in ["n", "no"]:
                            print("okay, exiting")
                            break
                    except:
                        print("Wrong input, try again")
                        continue
            else:
                print("All required models available")


from sentence_transformers import SentenceTransformer

model_path = (
    "/home/andreas/Documents/Python/local_llm/models/embeddings/gte-multilingual-base"
)

model = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base",
    trust_remote_code=True,
    cache_folder=model_path,  # downloads + caches here
)
