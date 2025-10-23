#!/bin/bash
export OLLAMA_MODELS="/home/andreas/Documents/Python/local_llm/models/llm/ollama"
export OLLAMA_KEEP_ALIVE=600    # keep the model loaded for 10 minutes
/home/andreas/Documents/Python/local_llm/models/llm/ollama_runtime/bin/ollama serve
