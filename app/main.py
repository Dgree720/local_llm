from llm_service import LLM_Service
from rag_service import HybridRetriever
import sys
from omegaconf import OmegaConf
import time
from rich.console import Console
from alive_progress import alive_bar
import requests

config = OmegaConf.load("config.yaml")


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
            time.sleep(0.005)
            bar()

    llm = LLM_Service(config.llm_config)

    with alive_bar(100) as bar:
        for _ in range(100):
            time.sleep(0.005)
            bar()

    ret = HybridRetriever(config.retriever_config)
    with alive_bar(100) as bar:
        for _ in range(100):
            time.sleep(0.005)
            bar()

    time.sleep(0.25)

    console.clear()

    console.print("[bold blue]Chat Started. Ask away![/bold blue]")
    console.print("[bold blue]=[/bold blue]" * 50)

    while True:
        user_query = console.input("[bold green]User: [/bold green]")

        retrieved_context = ret.retrieve(user_query)
        # start_time = time.time()
        console.print("[bold blue]\nAI Response[/bold blue]")
        llm.get_response(user_query, retrieved_context, console)
        print("\n")
        # end_time = time.time()
        # time_elapsed = end_time - start_time
        # print(f"\ntime: {time_elapsed}")


if __name__ == "__main__":
    main()
