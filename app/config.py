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

# 系統參數設定（支援環境變數覆寫）
def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# bool 解析
def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value_norm = value.strip().lower()
    if value_norm in ("1", "true", "yes", "y", "on"):
        return True
    if value_norm in ("0", "false", "no", "n", "off"):
        return False
    return default


# 語言模型最大輸出token數（OPENAI_MAX_TOKENS）
MAX_TOKENS = _get_env_int("OPENAI_MAX_TOKENS", 8196)

# 詞嵌入模型名稱（EMBEDDING_MODEL_NAME）
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")

# 檢索參數（皆可由環境變數覆蓋）
CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 100)  # 文本分塊大小
CHUNK_OVERLAP = _get_env_int("CHUNK_OVERLAP", 40)  # 文本分塊重疊大小
MAX_CHUNKS = _get_env_int("MAX_CHUNKS", 10)  # 最大分塊數量
MAX_RESULTS = _get_env_int("MAX_RESULTS", 70)  # 結果數量超過此值時進行進一步篩選
TOP_K = _get_env_int("TOP_K", 10)  # 相關性排序後選取的top k筆數

# API設定（支援環境變數覆寫）
_cors_env = os.getenv("CORS_ORIGINS")
CORS_ORIGINS = _cors_env.split(",") if _cors_env and _cors_env.strip() != "" else ["*"]
API_TITLE = os.getenv("API_TITLE", "RAG System API")  # API標題

# Reranker 設定（支援 Cross-Encoder）
RERANKER_USE_CROSS_ENCODER = _get_env_bool("RERANKER_USE_CROSS_ENCODER", False)
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLE_IN_PIPELINE = _get_env_bool("RERANKER_ENABLE_IN_PIPELINE", False)