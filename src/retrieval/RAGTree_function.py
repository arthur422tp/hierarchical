"""
這裡蒐集樹建立、儲存以及檢索用函式
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import torch
import faiss
import pickle
import multiprocessing as mp

from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI

from transformers import pipeline
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist

from collections import deque
import heapq
from typing import List, Tuple
import re

from fastcluster import linkage
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, cdist

import src.retrieval.generated_function as gf


##Tree方法函數

#Node建立
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
    nodes = [Node(vector / np.linalg.norm(vector), text, i) for i, (vector, text) in enumerate(zip(vectors, texts))]

    
    current_index = n

    for i, (c1, c2, dist, sample_count) in enumerate(linkage_matrix):
        c1, c2 = int(c1), int(c2)
        count_c1 = nodes[c1].sample_count
        count_c2 = nodes[c2].sample_count
        new_vector = (nodes[c1].vector * count_c1 + nodes[c2].vector * count_c2) / (count_c1 + count_c2)
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
    linkage_matrix = linkage(vectors, method='single', metric='cosine')
    root = build_tree(vectors, texts, linkage_matrix)
    return root

#rerank函數

def rerank_texts(query, passages, model_name="BAAI/bge-reranker-large"):
    """
    Summary:
    這是一個rerank函式

    query: str
    passages: list[str]
    model_name: 選定BAAI/bge-reranker-large(可更改)
    """
    reranker = FlagReranker(model_name, use_fp16=True)
    scores = reranker.compute_score([[query, passage] for passage in passages])
    
    scored_passages = sorted(zip(scores, passages), reverse=True)
    
    return [passage for score, passage in scored_passages]


def save_tree(root, filename):
    with open(filename, 'wb') as f:
        pickle.dump(root, f)
    print(f"Tree saved to {filename}")


def load_tree(filename):
    with open(filename, 'rb') as f:
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
        text_splitter = RecursiveCharacterTextSplitter(separators=["【"],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        chunked_texts = text_splitter.split_text(self.text)
        
        return chunked_texts[:max_chunks]

def find_most_similar_node(root, query, model):
    """
    使用BFS搜索檢索樹內最相似的節點。
    """
    min_distance = float('inf')
    most_similar_node = None
    query_vector = model.encode(query)

    stack = [root]
    while stack:
        node = stack.pop()
        distance = cosine(query_vector, node.vector)
        if distance < min_distance:
            min_distance = distance
            most_similar_node = node
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
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

def tree_search(root, query, model, chunk_size, chunk_overlap, max_chunks=10):
    """
    找尋最接近的文本。
    """
    results = set()
    queries = [query]
    
    
    if len(query) > chunk_size:
        qp = QueryProcessor(query)
        queries = qp.text_chunking(chunk_size, chunk_overlap, max_chunks)
    
    for sub_query in queries:
        retrieved_texts, retrieved_vectors = query_tree(root, sub_query, model)
        
        if len(retrieved_texts) > 70:
            query_vector = model.encode([sub_query])
            similarities = 1 - cdist(query_vector.reshape(1, -1), retrieved_vectors, metric="cosine")[0]
            top_indices = np.argsort(similarities)[-10:][::-1]
            refined_docs = [retrieved_texts[i] for i in top_indices]
            results.update(refined_docs)
        else:
            results.update(retrieved_texts)
    
    return list(results)

def extraction_tree_search(root, query, model, chunk_size, chunk_overlap, llm, max_chunks=10):
    """
    有進行query extraction的檢索法。
    """

    simplified_query = gf.query_extraction(query, llm)
    print(f"Simplified Query: {simplified_query}")
    
    results = set()
    queries = [simplified_query]
    

    if len(simplified_query) > chunk_size:
        qp = QueryProcessor(simplified_query)
        queries = qp.text_chunking(chunk_size, chunk_overlap, max_chunks)
    
 
    for sub_query in queries:
        retrieved_texts, retrieved_vectors = query_tree(root, sub_query, model)
        
        if len(retrieved_texts) > 50:
            query_vector = model.encode([sub_query])
            similarities = 1 - cdist(query_vector.reshape(1, -1), retrieved_vectors, metric="cosine")[0]
            top_indices = np.argsort(similarities)[-10:][::-1]
            refined_docs = [retrieved_texts[i] for i in top_indices]
            results.update(refined_docs)
        else:
            results.update(retrieved_texts)
    
    return list(results)
