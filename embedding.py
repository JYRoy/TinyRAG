import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BaseEmbeddings:
    """
    Base class for embeddings
    """

    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        """获取文本的向量表示的"""

        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """计算两个向量之间的余弦相似度"""
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """

    def __init__(
        self, path: str = "jinaai/jina-embeddings-v2-base-zh", is_api: bool = False
    ) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()
        self.path = path

    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()

    def load_model(self):
        import torch
        from transformers import AutoModel

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model
