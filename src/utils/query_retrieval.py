"""
一般faiss內積檢索函式
"""

import faiss
import src.utils.word_embedding as word_embedding


class Retrieval:
    def __init__(self, text, model, embedding_dim):
        self.text = text
        self.model = None
        self.embedding_dim = embedding_dim

    def build_index(self, embeddings):
        """
        建造向量資料庫
        """
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def retrieve(self, query, text, index, k: int):
        """
        檢索top-k
        """
        we = word_embedding.WordEmbedding()
        model = we.load_model()
        query_embedding = model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        _, I = index.search(query_embedding, k)  # noqa: E741
        retrieved_docs = [text[i] for i in I[0]]
        return retrieved_docs
