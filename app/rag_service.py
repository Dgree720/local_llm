from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
from omegaconf import OmegaConf

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

        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model_path)
        self.chroma = self.vectorize()

    def vectorize(self):
        if os.path.isdir(self.vector_path):
            chroma = Chroma(
                persist_directory=self.vector_path, embedding_function=self.embedder
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
                documents=chunks,
                persist_directory=self.vector_path,
                embedding_function=self.embedder,
            )

        return chroma

    def retrieve(self, query):
        retriever = self.chroma.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.6},
        )
        relevant_docs_raw = retriever.invoke(query)
        relevant_docs_raw
        relevant_docs = "".join(
            [Document.page_content for Document in relevant_docs_raw]
        )

        return relevant_docs
