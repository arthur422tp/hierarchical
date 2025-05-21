"""
做embedding用的函式
"""

import torch
from sentence_transformers import SentenceTransformer
import sys
import os

# 添加專案根目錄到路徑，以便引入其他模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 導入配置
from app.config import MODEL_NAME


class WordEmbedding:
    def __init__(self):
        self.model = None
        self.device = self._determine_device()
    
    def _determine_device(self):
        """
        確定最佳運算裝置
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        """
        加載模型用，只加載一次，避免重複加載
        """
        if self.model is None:
            print(f"正在載入模型 {MODEL_NAME} 到 {self.device} 裝置...")
            self.model = SentenceTransformer(MODEL_NAME)
            self.model = self.model.to(self.device)
            print(f"模型已成功載入到 {self.device} 裝置")

        return self.model

    def embedding(self, text):
        """
        做embedding用，支援批次處理
        """
        self.load_model()
        
        # 批次處理優化：如果是列表但長度為1，直接轉為單一文本
        if isinstance(text, list) and len(text) == 1:
            text = text[0]
            
        return self.model.encode(text, device=self.device)
