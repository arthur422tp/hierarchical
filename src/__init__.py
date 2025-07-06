"""
Hierarchical RAG Retrieval - AI-Powered Legal Document Retrieval Engine

這個套件提供基於階層式聚類與RAG技術的法律文件檢索引擎。
"""

__version__ = "0.1.0"

# 導入主要模組
from . import retrieval
from . import utils  
from . import data_processing

__all__ = ["retrieval", "utils", "data_processing"] 