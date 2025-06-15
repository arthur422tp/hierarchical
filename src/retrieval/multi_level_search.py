"""
多層索引檢索模組 - 提供與現有系統整合的接口
"""

import sys
import os
import numpy as np

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.multi_level_index import create_multi_level_index, query_multi_level_index
from src.utils.word_chunking import RagChunking
from app.config import MAX_RESULTS

class MultiLevelQueryProcessor:
    """
    多層索引查詢處理器
    """
    def __init__(self, text):
        self.text = text
        
    def text_chunking(self, chunk_size: int, chunk_overlap: int, max_chunks=10):
        """
        切短query用的函式
        """
        chunker = RagChunking(self.text)
        chunked_texts = chunker.text_chunking(chunk_size, chunk_overlap)
        return chunked_texts[:max_chunks]

def multi_level_tree_search(index, query: str, model, chunk_size: int, chunk_overlap: int, max_chunks: int = 10):
    """
    使用多層索引進行檢索
    
    Args:
        index: 多層索引
        query: 查詢字符串
        model: 詞嵌入模型
        chunk_size: 文本分塊大小
        chunk_overlap: 文本分塊重疊大小
        max_chunks: 最大分塊數量
        
    Returns:
        list: 檢索到的文本列表
    """
    # 處理查詢
    queries = [query]
    if len(query) > chunk_size:
        processor = MultiLevelQueryProcessor(query)
        queries = processor.text_chunking(chunk_size, chunk_overlap, max_chunks)
        
    # 收集結果
    results = set()
    for sub_query in queries:
        retrieved_texts = query_multi_level_index(
            index, 
            sub_query, 
            model, 
            top_k=3, 
            max_results=MAX_RESULTS
        )
        results.update(retrieved_texts)
        
    return list(results)

def multi_level_extraction_tree_search(
    index, query: str, model, chunk_size: int, chunk_overlap: int, llm, max_chunks: int = 10
):
    """
    使用查詢提取和多層索引進行檢索
    
    Args:
        index: 多層索引
        query: 查詢字符串
        model: 詞嵌入模型
        chunk_size: 文本分塊大小
        chunk_overlap: 文本分塊重疊大小
        llm: 語言模型
        max_chunks: 最大分塊數量
        
    Returns:
        list: 檢索到的文本列表
    """
    # 導入查詢提取函數
    from src.retrieval.generated_function import GeneratedFunction
    
    # 提取查詢
    generator = GeneratedFunction()
    simplified_query = generator.query_extraction(query, llm)
    
    # 處理查詢
    queries = [simplified_query]
    if len(simplified_query) > chunk_size:
        processor = MultiLevelQueryProcessor(simplified_query)
        queries = processor.text_chunking(chunk_size, chunk_overlap, max_chunks)
        
    # 收集結果
    results = set()
    for sub_query in queries:
        retrieved_texts = query_multi_level_index(
            index, 
            sub_query, 
            model, 
            top_k=3, 
            max_results=MAX_RESULTS
        )
        results.update(retrieved_texts)
        
    return list(results)

def build_multi_level_index_from_files(embeddings_path, texts_path):
    """
    從文件構建多層索引
    
    Args:
        embeddings_path: 向量文件路徑
        texts_path: 文本文件路徑
        
    Returns:
        MultiLevelIndex: 構建的索引
    """
    import pickle
    
    # 加載向量和文本
    with open(embeddings_path, 'rb') as f:
        vectors = pickle.load(f)
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
        
    # 確保向量是numpy數組
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # 構建索引
    return create_multi_level_index(vectors, texts)