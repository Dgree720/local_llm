from sentence_transformers import SentenceTransformer

model_path = (
    "/home/andreas/Documents/Python/local_llm/models/embeddings/gte-multilingual-base"
)
model = SentenceTransformer(model_path, trust_remote_code=True)
print("✅ Embedding dimension:", model.get_sentence_embedding_dimension())

texts = [
    "The Transformer architecture changed NLP.",
    "Die Transformer-Architektur revolutionierte NLP.",
]
embs = model.encode(texts, normalize_embeddings=True)
print(embs.shape)  # (2, 768)
