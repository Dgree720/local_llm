from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from transformers import logging
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
from omegaconf import OmegaConf

logging.set_verbosity_error()

config = OmegaConf.load("config.yaml")


class Retriever:
    def __init__(self, retriever_config: OmegaConf):
        self.config = retriever_config
        self.embedding_model_name = retriever_config.embed_model
        self.embedding_model_path = retriever_config.embedding_model_path
        self.vector_path = retriever_config.vector_db_path
        self.knowledge_path = retriever_config.knowledge_path
        self.chunk_size = retriever_config.chunk_size
        self.chunk_overlap = retriever_config.chunk_overlap

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )

        # self.chroma = self.vectorize()

        # self.retriever = self.chroma.as_retriever(
        #    search_type="similarity_score_threshold",
        #    search_kwargs={"k": 3, "score_threshold": 0.5},
        # )

        self.faiss_store = self.faiss_store()

    def faiss_store(self):
        loader = DirectoryLoader(self.knowledge_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", " "],
        )
        chunks = splitter.split_documents(docs)

        faiss = FAISS.from_documents(documents=chunks, embedding=self.embedder)

        return faiss

    def vectorize(self):
        if os.path.isdir(self.vector_path):
            chroma = Chroma(
                persist_directory=self.vector_path, embedding_function=self.embedder
            )

            return chroma
        else:
            os.makedirs(self.vector_path, exist_ok=True)

            loader = DirectoryLoader(self.knowledge_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", " "],
            )
            chunks = splitter.split_documents(docs)

            for i, doc in enumerate(chunks):
                doc.metadata["chunk_id"] = i

            chroma = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedder,
                persist_directory=self.vector_path,
            )

            return chroma

    def retrieve(self, query):
        relevant_docs_raw = self.retriever.invoke(query)
        relevant_docs = ""
        for i, doc in enumerate(relevant_docs_raw):
            relevant_docs += f"{doc.page_content}\n"

        return relevant_docs

    def retrieve_faiss(self, query):
        relevant_docs_raw = self.faiss_store.similarity_search(query, k=5)
        relevant_docs = ""
        for i, doc in enumerate(relevant_docs_raw):
            relevant_docs += f"{doc.page_content}\n"

        return relevant_docs


"""
rag = Retriever(config.retriever_config)
docs = rag.retrieve_faiss("What is a transformer")
print(f"\nretrieved: \n{docs}")
"""
