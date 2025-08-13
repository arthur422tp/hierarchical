"""
這裡蒐集樹建立、儲存以及檢索用函式
"""

import numpy as np
import pickle
import sys
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 添加專案根目錄到路徑，以便引入其他模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import MAX_RESULTS, TOP_K
from app.config import RERANKER_USE_CROSS_ENCODER, RERANKER_MODEL_NAME
from app.config import RERANKER_ENABLE_IN_PIPELINE

from fastcluster import linkage
from collections import deque

from scipy.spatial.distance import cosine, cdist
from sklearn.preprocessing import normalize
import torch
from sentence_transformers import CrossEncoder

import src.retrieval.generated_function as gf


##Tree方法函數


# Node建立
class Node:
    def __init__(self, vector, text=None, index=None, left=None, right=None):
        self.vector = vector
        self.text = text
        self.index = index
        self.left = left
        self.right = right
        self.sample_count = 1
        self.subtree_depth = 0


def calculate_subtree_depth(node):
    """
    Summary:
    用來計算樹的深度的函式

    node:節點
    """
    if node is None:
        return -1
    left_depth = calculate_subtree_depth(node.left)
    right_depth = calculate_subtree_depth(node.right)
    node.subtree_depth = max(left_depth, right_depth) + 1
    return node.subtree_depth


def build_tree(vectors, texts, linkage_matrix):
    """
    Summary:
    這是關於檢索樹建構的演算法

    vectors: np.array
    texts: list[str]
    linkage_matrix: 可自訂
    """
    n = len(vectors)
    nodes = [
        Node(vector / np.linalg.norm(vector), text, i)
        for i, (vector, text) in enumerate(zip(vectors, texts))
    ]

    current_index = n

    for i, (c1, c2, dist, sample_count) in enumerate(linkage_matrix):
        c1, c2 = int(c1), int(c2)
        count_c1 = nodes[c1].sample_count
        count_c2 = nodes[c2].sample_count
        new_vector = (nodes[c1].vector * count_c1 + nodes[c2].vector * count_c2) / (
            count_c1 + count_c2
        )
        new_vector /= np.linalg.norm(new_vector)

        new_node = Node(new_vector, None, current_index, nodes[c1], nodes[c2])
        new_node.sample_count = count_c1 + count_c2
        nodes.append(new_node)

        current_index += 1

    root = nodes[-1]
    calculate_subtree_depth(root)
    return root


def create_ahc_tree(vectors, texts):
    """
    Summary:
    建構檢索樹

    vectors: np.array
    texts: list[str]
    """
    linkage_matrix = linkage(vectors, method="single", metric="cosine")
    root = build_tree(vectors, texts, linkage_matrix)
    return root


# rerank函數


_CROSS_ENCODER_INSTANCE = None


def _determine_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_cross_encoder():
    global _CROSS_ENCODER_INSTANCE
    if _CROSS_ENCODER_INSTANCE is None:
        device = _determine_device()
        _CROSS_ENCODER_INSTANCE = CrossEncoder(RERANKER_MODEL_NAME, device=device)
        print(f"已載入 CrossEncoder: {RERANKER_MODEL_NAME} 到 {device}")
    return _CROSS_ENCODER_INSTANCE


def rerank_texts(query, passages, model, k):
    """
    Summary:
    可切換的文本重排序函數：
    - 若啟用 Cross-Encoder，使用 cross-encoder 直接對 (query, passage) 配對打分
    - 否則回退為 embedding 餘弦相似度排序

    query: str
    passages: list[str]
    model: sentence-transformers embedding model（在未啟用 Cross-Encoder 時使用）
    k: int
    """
    if RERANKER_USE_CROSS_ENCODER:
        ce = _get_cross_encoder()
        pairs = [(query, p) for p in passages]
        scores = ce.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [passages[i] for i in ranked_indices]

    # 回退：使用 embedding 餘弦相似度
    query_vector = model.encode(query)
    passage_vectors = model.encode(passages)

    query_vector = normalize(query_vector)
    passage_vectors = normalize(passage_vectors)

    similarities = np.dot(passage_vectors, query_vector.T).squeeze()
    ranked_indices = np.argsort(similarities)[::-1][:k]
    return [passages[i] for i in ranked_indices]


def save_tree(root, filename):
    with open(filename, "wb") as f:
        pickle.dump(root, f)
    print(f"Tree saved to {filename}")


def load_tree(filename):
    with open(filename, "rb") as f:
        root = pickle.load(f)
    print(f"Tree loaded from {filename}")
    return root


class QueryProcessor:
    def __init__(self, text):
        self.text = text

    def text_chunking(self, chunk_size: int, chunk_overlap: int, max_chunks=10):
        """
        切短query用的函式
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["【"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunked_texts = text_splitter.split_text(self.text)

        return chunked_texts[:max_chunks]


def find_most_similar_node(root, query, model):
    """
    使用BFS搜索檢索樹內最相似的節點。
    """
    min_distance = float("inf")
    most_similar_node = None
    query_vector = model.encode(query)

    queue = deque([root])
    while queue:
        node = queue.popleft()
        distance = cosine(query_vector, node.vector)
        if distance < min_distance:
            min_distance = distance
            most_similar_node = node
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return most_similar_node


def collect_leaf_texts(node):
    """
    回傳所蒐集到的文本。
    """
    if node.left is None and node.right is None:
        return [node.text], [node.vector]

    texts, vectors = [], []
    if node.left:
        left_texts, left_vectors = collect_leaf_texts(node.left)
        texts.extend(left_texts)
        vectors.extend(left_vectors)
    if node.right:
        right_texts, right_vectors = collect_leaf_texts(node.right)
        texts.extend(right_texts)
        vectors.extend(right_vectors)

    return texts, vectors


def query_tree(root, query, model):
    most_similar_node = find_most_similar_node(root, query, model)
    if most_similar_node.left is None and most_similar_node.right is None:
        return [most_similar_node.text], [most_similar_node.vector]
    else:
        return collect_leaf_texts(most_similar_node)


def _process_retrieved_texts(retrieved_texts, retrieved_vectors, sub_query, model):
    """
    處理檢索到的文本，如果超過70個文本則進行排序和篩選。
    
    Args:
        retrieved_texts: 檢索到的文本列表
        retrieved_vectors: 檢索到的文本向量列表
        sub_query: 子查詢
        model: 詞嵌入模型
        
    Returns:
        list: 處理後的文本列表
    """
    if len(retrieved_texts) > MAX_RESULTS:
        # 對於大量檢索結果，做進一步排序篩選
        if RERANKER_ENABLE_IN_PIPELINE:
            try:
                # 使用通用的 rerank_texts（可根據環境切換 Cross-Encoder / embedding）
                reranked = rerank_texts(sub_query, retrieved_texts, model, TOP_K)
                return reranked
            except Exception as _:
                # 回退到向量相似度 rerank
                pass

        query_vector = model.encode([sub_query])
        similarities = (
            1 - cdist(query_vector.reshape(1, -1), retrieved_vectors, metric="cosine")[0]
        )
        # 取前TOP_K個最相關的
        top_indices = np.argsort(similarities)[-TOP_K:][::-1]
        return [retrieved_texts[i] for i in top_indices]
    else:
        # 對於少量結果，直接返回
        return retrieved_texts


def _process_queries(queries, root, model, max_chunks=10):
    """
    處理多個查詢並合併結果
    
    Args:
        queries: 查詢列表
        root: 檢索樹根節點
        model: 詞嵌入模型
        max_chunks: 最大塊數
        
    Returns:
        list: 合併後的不重複文本列表
    """
    results = set()
    
    for sub_query in queries:
        retrieved_texts, retrieved_vectors = query_tree(root, sub_query, model)
        processed_texts = _process_retrieved_texts(retrieved_texts, retrieved_vectors, sub_query, model)
        results.update(processed_texts)
        
    return list(results)


def tree_search(root, query: str, model, chunk_size: int, chunk_overlap: int, max_chunks: int = 10):
    """
    找尋最接近的文本。
    
    Args:
        root: 檢索樹根節點
        query: 查詢字符串
        model: 詞嵌入模型
        chunk_size: 文本分塊大小
        chunk_overlap: 文本分塊重疊大小
        max_chunks: 最大分塊數量
        
    Returns:
        list: 檢索到的文本列表
    """
    queries = [query]

    if len(query) > chunk_size:
        qp = QueryProcessor(query)
        queries = qp.text_chunking(chunk_size, chunk_overlap, max_chunks)

    return _process_queries(queries, root, model, max_chunks)


def extraction_tree_search(
    root, query: str, model, chunk_size: int, chunk_overlap: int, llm, max_chunks: int = 10
):
    """
    有進行query extraction的檢索法。
    
    Args:
        root: 檢索樹根節點
        query: 原始查詢字符串
        model: 詞嵌入模型
        chunk_size: 文本分塊大小
        chunk_overlap: 文本分塊重疊大小
        llm: 語言模型
        max_chunks: 最大分塊數量
        
    Returns:
        list: 檢索到的文本列表
    """
    generator = gf.GeneratedFunction()
    simplified_query = generator.query_extraction(query, llm)
    
    queries = [simplified_query]

    if len(simplified_query) > chunk_size:
        qp = QueryProcessor(simplified_query)
        queries = qp.text_chunking(chunk_size, chunk_overlap, max_chunks)

    return _process_queries(queries, root, model, max_chunks)
