from langchain_ollama import OllamaLLM
import subprocess
import time
from pathlib import Path
import stat


def fetch_ollama():
    script = Path("/home/andreas/Documents/local_llm/serve_ollama.sh")

    try:
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        subprocess.Popen(
            ["bash", "/home/andreas/Documents/local_llm/serve_ollama.sh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("ollama starting")
        time.sleep(3)
    except:
        print("error occurred")

    ollama = OllamaLLM(model="gemma3:1b", temperature=0.7)

    return ollama


llm = fetch_ollama()

llm.invoke("hi")
