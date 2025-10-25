import os
from llama_cpp import Llama
import time


llm = Llama(
    model_path="/home/andreas/Documents/Python/local_llm/llama_cpp/models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_gpu_layers=99,
    n_ctx=32768,
    verbose=False,
)

start = time.time()
output = llm.cre("Explain LLM transformers to me in detail.", max_tokens=2000)
text = output["choices"][0]["text"]
print(text)
end = time.time()
elapsed = end - start
words = len(text)

print(f"created {words} words in {elapsed} time")
