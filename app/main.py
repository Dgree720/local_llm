from llm_service import LLM_Service
from rag_service import Retriever
import sys
from omegaconf import OmegaConf
import os
import time
from rich.console import Console
from alive_progress import alive_bar
import requests


os.environ["OLLAMA_MODELS"] = (
    "/home/andreas/Documents/Python/local_llm/models/llm/ollama"
)


def main():
    console = Console()
    console.clear()
    user_os = sys.platform
    config = OmegaConf.load("config.yaml")

    console.print("[bold blue]Welcome![/bold blue]\n")
    print(
        f"User system recognized as using {user_os} as OS. Setting things up accordingly...\n"
    )
    with alive_bar(100) as bar:
        for _ in range(100):
            time.sleep(0.01)
            bar()

    llm = LLM_Service()

    if requests.get("http://127.0.0.1:11434/api/") == 200:
        print("ollama already running")
    else:
        llm.start_ollama_server()

    llm.fetch_llm()
    llm.make_chain()

    with alive_bar(100) as bar:
        for _ in range(100):
            time.sleep(0.01)
            bar()
    print(f"LLM {llm.llm.model} running\n")

    ret = Retriever(config.retriever_config)
    with alive_bar(100) as bar:
        for _ in range(100):
            time.sleep(0.01)
            bar()

    print(f"Retriever {ret.embedding_model_name} set up\n")

    time.sleep(0.75)

    console.clear()

    console.print("[bold blue]Chat Started. Ask away![/bold blue]")
    console.print("[bold blue]=[/bold blue]" * 50)

    while True:
        user_query = console.input("[bold green]User: [/bold green]")

        retrieved_context = ret.retrieve(user_query)

        start_time = time.time()
        llm.get_quick_response(user_query, retrieved_context)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(time_elapsed)


if __name__ == "__main__":
    main()
