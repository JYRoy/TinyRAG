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


class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(
        self, path: str = "BAAI/bge-base-zh-v1.5", is_api: bool = False
    ) -> None:
        super().__init__(path, is_api)
        self._model, self._tokenizer = self.load_model(path)

    def get_embedding(self, text: str) -> List[float]:
        import torch

        encoded_input = self._tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self._model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings[0].tolist()

    def load_model(self, path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        model.eval()
        return model, tokenizer


class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """

    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI

            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text,
        )
        return response.data[0].embedding
