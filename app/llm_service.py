from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import subprocess
import time
from pathlib import Path
import stat
import os
import sys
from prompts import make_prompt
import requests as re
import json


class LLM_Service:
    def __init__(self):
        os.environ["OLLAMA_MODELS"] = (
            "/home/andreas/Documents/Python/local_llm/models/llm/ollama"
        )
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

        # self.server = self.start_ollama_server()
        # self.llm = self.fetch_llm()
        # self.prompt = make_prompt()
        # self.chain = self.make_chain()

    def start_ollama_server(self):
        # check if instance of ollama server is already running

        print("Starting Ollama server")
        user_os = sys.platform

        if user_os in ["linux", "darwin"]:
            # print(f"User OS recognized as {user_os}")
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
                time.sleep(5)
                return True
            except Exception as e:
                print(f"error occurred: {e}")
                return False

        elif user_os in ["win32", "cygwin"]:
            script = Path("/home/andreas/Documents/local_llm/serve_ollama_windows.ps1")

            try:
                process = subprocess.Popen(
                    [
                        "powershell.exe",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        script,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                print("ollama starting")
                time.sleep(5)
                return True

            except Exception as e:
                print(f"error occurred: {e}")
                return False

        else:
            print(f"Operating System {os} not supported at the moment")
            return False

    def fetch_llm(self):
        llm = OllamaLLM(model="gemma3:1b", temperature=0.4)

        self.llm = llm
        # print(f"LLM {llm.model} running")

        return llm

    def make_chain(self):
        prompt = PromptTemplate(
            template=self.prompt, input_variables=["user_query", "context"]
        )
        chain = (
            {
                "user_query": RunnablePassthrough(),
                "context": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def get_response(self, user_query, context):
        response = self.chain.invoke({"user_query": user_query, "context": context})

        return response

    def get_quick_response(self, user_query, context):
        prompt = make_prompt(user_query, context)
        print(prompt)
        url = "http://127.0.0.1:11434/api/generate"
        data = {"model": "gemma3:1b", "prompt": prompt, "stream": True}

        with re.post(url, json=data, stream=True) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                if "response" in chunk:
                    sys.stdout.write(chunk["response"])
                    sys.stdout.flush()
                if chunk.get("done"):
                    break
        print()

    def stream_response(self, user_query, context):
        for chunk in self.llm.stream("Why do parrots have colorful feathers?"):
            print(chunk.text, end="|", flush=True)
