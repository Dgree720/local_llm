from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from prompts import make_prompt
from omegaconf import OmegaConf
from rich.console import Console
from rich.markdown import Markdown

config = OmegaConf.load("config.yaml")


class LLM_Service:
    def __init__(self, llm_config: OmegaConf):
        self.config = llm_config
        self.model_path = llm_config.model_path
        self.llm = self.fetch_llm()
        self.prompt = make_prompt()
        self.chain = self.make_chain()

    def fetch_llm(self):
        llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=self.config.get("n_ctx", 32768),
            n_batch=self.config.get("n_batch", 512),
            n_gpu_layers=self.config.get("n_gpu_layers", 0),  # Use GPU if available
            temperature=self.config.get("temperature", 0.2),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 40),
            max_tokens=self.config.get("max_tokens", 512),
            repeat_penalty=self.config.get("repeat_penalty", 1.1),
            verbose=self.config.get("verbose", False),
            streaming=True,  # Enable streaming by default
        )

        return llm

    def make_chain(self):
        prompt = PromptTemplate(
            template=self.prompt, input_variables=["user_query", "context"]
        )
        self.chain = (
            {
                "user_query": lambda x: x["user_query"],
                "context": lambda x: x.get("context", "No context provided."),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return self.chain

    def stream_response(self, user_query: str, context: str | None = None):
        for chunk in self.chain.stream({"user_query": user_query, "context": context}):
            yield chunk

    def get_response(self, user_query, context, console: Console):
        for chunk in self.chain.stream({"user_query": user_query, "context": context}):
            if chunk is not None:
                if isinstance(chunk, dict) and "text" in chunk:
                    print(chunk["text"], end="", flush=True)
                else:
                    print(chunk, end="", flush=True)
