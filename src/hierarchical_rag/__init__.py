"""
Hierarchical RAG Retrieval - AI-Powered Legal Document Retrieval Engine

這個套件提供基於階層式聚類與RAG技術的法律文件檢索引擎。
核心功能包括：
- 階層式聚類檢索樹建構
- 多層索引檢索
- 文本向量化與相似度計算
- 查詢提取與處理
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 導入主要類別和函數
from .retrieval.RAGTree_function import (
    Node,
    create_ahc_tree,
    build_tree,
    tree_search,
    extraction_tree_search,
    find_most_similar_node,
    collect_leaf_texts,
    rerank_texts,
    save_tree,
    load_tree,
    QueryProcessor
)

from .retrieval.multi_level_search import (
    MultiLevelQueryProcessor,
    multi_level_tree_search,
    multi_level_extraction_tree_search,
    build_multi_level_index_from_files
)

from .utils.word_embedding import WordEmbedding
from .utils.word_chunking import RagChunking
from .utils.query_retrieval import Retrieval

from .data_processing.data_dealer import DataDealer

# 定義公開的API
__all__ = [
    # 核心檢索類別
    "Node",
    "QueryProcessor", 
    "MultiLevelQueryProcessor",
    "WordEmbedding",
    "RagChunking",
    "Retrieval",
    "DataDealer",
    
    # 檢索樹函數
    "create_ahc_tree",
    "build_tree", 
    "tree_search",
    "extraction_tree_search",
    "find_most_similar_node",
    "collect_leaf_texts",
    "rerank_texts",
    "save_tree",
    "load_tree",
    
    # 多層索引函數
    "multi_level_tree_search",
    "multi_level_extraction_tree_search", 
    "build_multi_level_index_from_files",
]

# 版本資訊
def get_version():
    """返回套件版本"""
    return __version__

def get_info():
    """返回套件基本資訊"""
    return {
        "name": "hierarchical-rag-retrieval",
        "version": __version__,
        "author": __author__,
        "description": "AI-Powered Legal Document Retrieval Engine based on Hierarchical Clustering & RAG"
    } 