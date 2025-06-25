"""
檢索模組 - 提供階層式聚類檢索和多層索引檢索功能
"""

from .RAGTree_function import (
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

from .multi_level_search import (
    MultiLevelQueryProcessor,
    multi_level_tree_search,
    multi_level_extraction_tree_search,
    build_multi_level_index_from_files
)

from .generated_function import GeneratedFunction

__all__ = [
    # 核心檢索類別
    "Node",
    "QueryProcessor", 
    "MultiLevelQueryProcessor",
    "GeneratedFunction",
    
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
