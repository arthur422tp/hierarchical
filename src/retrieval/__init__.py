"""
檢索模組 - 提供階層式聚類檢索
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



from .generated_function import GeneratedFunction

__all__ = [
    # 核心檢索類別
    "Node",
    "QueryProcessor", 
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

]
