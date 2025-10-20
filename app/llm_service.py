from langchain_ollama import OllamaLLM
import subprocess
import time
from pathlib import Path
import stat
import os
import sys


def start_ollama_server():
    user_os = sys.platform()

    if user_os in ["linux", "darwin"]:
        script = Path("/home/andreas/Documents/local_llm/serve_ollama_posix.sh")
        try:
            script.chmod(script.stat().st_mode | stat.S_IEXEC)

            subprocess.Popen(
                ["bash", "/home/andreas/Documents/local_llm/serve_ollama_posix.sh"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("ollama starting")
            time.sleep(3)
        except:
            print("error occurred")

    elif user_os in ["win32", "cygwin"]:
        script = Path("/home/andreas/Documents/local_llm/serve_ollama_windows.ps1")
        process = subprocess.Popen(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Read the output
        stdout, stderr = process.communicate()

        print("ollama starting")
        time.sleep(3)

    else:
        print(f"Operating System {os} not supported at the moment")


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
