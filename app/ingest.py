from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil

knowledge_path = "/home/andreas/Documents/Python/local_llm/documents"

# if os.path.isdir(CHROMA_DB_DIR):
#    shutil.rmtree(CHROMA_DB_DIR)

CHUNK_SIZE = 500
OVERLAP = 100
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")

embedder = HuggingFaceEmbeddings(
    model_name="/home/andreas/Documents/Python/local_llm/models/ibm-granite/granite-embedding-107m-multilingual"
)

if os.path.isdir(CHROMA_DB_DIR):
    chroma = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedder)
else:
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    loader = DirectoryLoader(knowledge_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
    )
    chunks = splitter.split_documents(docs)

    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i

    chroma = Chroma.from_documents(
        documents=chunks, persist_directory=CHROMA_DB_DIR, embedding=embedder
    )


retriever = chroma.as_retriever(search_kwargs={"k": 3})


relevant_docs = retriever.invoke("What is a transformer?")
