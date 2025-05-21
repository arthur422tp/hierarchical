"""
系統配置文件
"""
import os
from pathlib import Path

# 基礎路徑設定
APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = PROJECT_ROOT / "data" / "data_processed"

# 系統參數設定
MAX_TOKENS = 8196  # 語言模型最大輸出token數
MODEL_NAME = "intfloat/multilingual-e5-large"  # 詞嵌入模型名稱

# 檢索參數
CHUNK_SIZE = 100  # 文本分塊大小
CHUNK_OVERLAP = 40  # 文本分塊重疊大小
MAX_CHUNKS = 10  # 最大分塊數量
MAX_RESULTS = 70  # 結果數量超過此值時進行進一步篩選
TOP_K = 10  # 相關性排序後選取的top k筆數

# API設定
CORS_ORIGINS = ["*"]  # CORS設定
API_TITLE = "RAG System API"  # API標題 