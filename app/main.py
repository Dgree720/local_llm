import requests
from utils.download_models import download_models
from utils.configuration import config
from prompts import make_prompt
from llm_service import LLM_Service
from rag_service import Retriever
import sys
from omegaconf import OmegaConf


def main():
    user_os = sys.platform
    config = OmegaConf.load("config.yaml")

    print("Welcome!")
    print(
        f"User system recognized as using {user_os} as OS. Setting things up accordingly..."
    )

    llm = LLM_Service()
    ret = Retriever(config.retriever_config)
    user_query = input("input: ")
    retrieved_context = ret.retrieve(user_query)

    response = llm.get_response(user_query, retrieved_context)

    print(response)


if __name__ == "__main__":
    main()
