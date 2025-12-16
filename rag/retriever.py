import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class FaissRetriever:
    def __init__(self, documents):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(documents)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.documents = documents

    def retrieve(self, query, top_k=3):
        q_emb = self.model.encode([query])
        _, idx = self.index.search(q_emb, top_k)
        return [self.documents[i] for i in idx[0]]