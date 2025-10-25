from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import llamacpp

import subprocess
import time
from pathlib import Path
import stat
import os
import sys
from prompts import make_prompt, make_prompt2
import requests as re
import json


class LLM_Service:
    def __init__(self):
        self.llm = llamacpp.LlamaCpp(
            model_path="/home/andreas/Documents/Python/local_llm/llama_cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=4096,
            verbose=False,
        )

        self.prompt = make_prompt()

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

        self.chain = chain
        return chain

    def get_response(self, user_query, context):
        for chunk in self.chain.stream({"user_query": user_query, "context": context}):
            print(chunk, end="", flush=True)
