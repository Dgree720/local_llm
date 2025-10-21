from langchain_ollama import OllamaLLM
import subprocess
import time
from pathlib import Path
import stat
import os
import sys
from prompts

def start_ollama_server():
    user_os = sys.platform

    if user_os in ["linux", "darwin"]:
        print(f"User OS recognized as {user_os}")
        script = Path(
            "/home/andreas/Documents/Python/local_llm/app/serve_ollama_posix.sh"
        )
        try:
            script.chmod(script.stat().st_mode | stat.S_IEXEC)

            subprocess.Popen(
                [
                    "bash",
                    "/home/andreas/Documents/Python/local_llm/app/serve_ollama_posix.sh",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("ollama starting")
            time.sleep(3)
        except Exception as e:
            print(f"error occurred: {e}")

    elif user_os in ["win32", "cygwin"]:
        script = Path("/home/andreas/Documents/local_llm/serve_ollama_windows.ps1")

        try:
            process = subprocess.Popen(
                ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            print("ollama starting")
            time.sleep(3)

        except Exception as e:
            print(f"error occurred: {e}")

    else:
        print(f"Operating System {os} not supported at the moment")


def fetch_llm():
    llm = OllamaLLM(model="gemma3:1b", temperature=0.4)

    return llm
    

#llm = fetch_llm()

#llm.invoke("hi")
