from llm_service import LLM_Service
from rag_service import Retriever
import sys
from omegaconf import OmegaConf
import os
import time

os.environ["OLLAMA_MODELS"] = (
    "/home/andreas/Documents/Python/local_llm/models/llm/ollama"
)


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
    retrieved_context = "You are a large language model"  # ret.retrieve(user_query)
    # print(retrieved_context)

    start_time = time.time()
    response = llm.get_response(user_query, retrieved_context)

    print(response)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(time_elapsed)


if __name__ == "__main__":
    main()
