from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list[str]):
        return self.model.encode(texts)
    def compute_similarity(self, embeddings1, embeddings2):
        return util.cos_sim(embeddings1, embeddings2)
