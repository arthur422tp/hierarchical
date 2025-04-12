"""
做embedding用的函式
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class WordEmbedding():
    def __init__(self):
        self.model = None

    def load_model(self):
        """
        加載模型用
        """
        if self.model is None:
            self.model = SentenceTransformer('intfloat/multilingual-e5-large')

            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        return self.model.to(self.device)

    def embedding(self, text):
        """
        做embedding用
        """
        self.load_model()
        return self.model.encode(text, device=self.device)
