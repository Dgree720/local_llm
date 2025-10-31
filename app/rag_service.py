import os
import re
import faiss
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from transformers import logging
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pathlib import Path

logging.set_verbosity_error()
config = OmegaConf.load("config.yaml")


class HybridRetriever:
    def __init__(self, retriever_config: OmegaConf):
        self.config = retriever_config
        self.knowledge_path = Path(retriever_config.knowledge_path)
        self.vector_path = Path(retriever_config.vector_db_path)
        self.chunk_size = retriever_config.chunk_size
        self.chunk_overlap = retriever_config.chunk_overlap
        self.weight_faiss = retriever_config.get("semantic_weight", 0.6)
        self.weight_bm25 = retriever_config.get("bm25_weight", 0.4)

        self.embedder = HuggingFaceEmbeddings(
            model_name=retriever_config.embedding_model_path,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_path.mkdir(parents=True, exist_ok=True)
        self.faiss_store = self._load_or_create_faiss()
        self.bm25_retriever = self._load_bm25()

    # -----------------------------------------------------------
    # Setup
    # -----------------------------------------------------------
    def _load_or_create_faiss(self):
        faiss_index_path = self.vector_path / "faiss_index"
        if faiss_index_path.exists():
            print(f"📂 Loading FAISS index from {faiss_index_path}")
            return FAISS.load_local(
                str(faiss_index_path),
                self.embedder,
                allow_dangerous_deserialization=True,
            )
        print("🆕 Creating new FAISS index...")
        dim = len(self.embedder.embed_query("test"))
        index = faiss.IndexFlatIP(dim)
        return FAISS(
            embedding_function=self.embedder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def _load_bm25(self):
        print("📘 Loading BM25 retriever...")
        loader = DirectoryLoader(
            str(self.knowledge_path),
            glob="**/*.txt",
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        if not docs:
            raise RuntimeError(f"No documents found in {self.knowledge_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        print(f"BM25 initialized with {len(chunks)} chunks")
        return BM25Retriever.from_documents(chunks)

    # -----------------------------------------------------------
    # Add / Save documents for FAISS
    # -----------------------------------------------------------
    def add_documents_to_faiss(self):
        loader = DirectoryLoader(
            str(self.knowledge_path),
            glob="**/*.txt",
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        if not docs:
            print(f"⚠️ No docs in {self.knowledge_path}")
            return
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        for i, doc in enumerate(chunks):
            doc.metadata["chunk_id"] = i
        self.faiss_store.add_documents(chunks)
        self.faiss_store.save_local(str(self.vector_path / "faiss_index"))
        print("✅ FAISS index saved.")

    # -----------------------------------------------------------
    # Query Filtering
    # -----------------------------------------------------------
    @staticmethod
    def is_meaningful_query(query: str) -> bool:
        return len(query.strip()) > 3 and not re.fullmatch(
            r"^(hi|hello|hey|ok|thanks?|test)$", query.strip(), re.I
        )

    # -----------------------------------------------------------
    # Hybrid Retrieval
    # -----------------------------------------------------------
    def retrieve(self, query: str, k: int = 3, score_threshold: float = 0.55):
        if not self.is_meaningful_query(query):
            print("Generic query, skipping retrieval.")
            return []

        # BM25 Retrieval
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_scores = {
            doc.page_content: 1.0 - (i / len(bm25_docs))
            for i, doc in enumerate(bm25_docs)
        }

        # Semantic Retrieval
        faiss_docs_scores = self.faiss_store.similarity_search_with_relevance_scores(
            query, k=k
        )
        faiss_scores = {
            doc.page_content: score
            for doc, score in faiss_docs_scores
            if score >= score_threshold
        }

        # Combine
        combined = {}
        for content, score in faiss_scores.items():
            combined[content] = self.weight_faiss * score
        for content, score in bm25_scores.items():
            combined[content] = combined.get(content, 0) + self.weight_bm25 * score

        # Sort by combined score
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        print(f"\nHybrid retriever found {len(sorted_results)} results")
        return [
            Document(page_content=content, metadata={"score": score})
            for content, score in sorted_results[:k]
        ]


# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------
if __name__ == "__main__":
    hybrid = HybridRetriever(config.retriever_config)

    # Build FAISS once if missing
    if not (hybrid.vector_path / "faiss_index").exists():
        hybrid.add_documents_to_faiss()

    query = "Explain transformer architecture"
    results = hybrid.retrieve(query, k=5)

    for i, doc in enumerate(results):
        print(
            f"\nResult {i + 1} (score={doc.metadata['score']:.3f}):\n{doc.page_content[:500]}"
        )
