from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")


class Retriever:
    def __init__(self, retriever_config: OmegaConf):
        self.config = retriever_config
        self.model_name = retriever_config.embed_model
        self.vector_path = retriever_config.vector_store_path
        self.embedding_model_path = retriever_config.embedding_model_path
        self.knowledge_path = retriever_config.knowledge_path
        self.chunk_size = retriever_config.chunk_size
        self.chunk_overlap = retriever_config.chunk_overlap

        embedder = HuggingFaceEmbeddings(model_name=self.embedding_model_path)

    def vectorize(self):
        if os.path.isdir(self.vector_path):
            chroma = Chroma(
                persist_directory=self.vector_path, embedding_function=embedder
            )
        else:
            os.makedirs(self.vector_path, exist_ok=True)

            loader = DirectoryLoader(self.knowledge_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents(docs)

            for i, doc in enumerate(chunks):
                doc.metadata["chunk_id"] = i

            chroma = Chroma.from_documents(
                documents=chunks, persist_directory=self.vector_path, embedding=embedder
            )

    def retrieve(self, query):
        retriever = chroma.as_retriever(search_kwargs={"k": 3})


relevant_docs = retriever.invoke("What is a transformer?")

ret = Retriever(config.retriever_config)
print(ret.model_name)
