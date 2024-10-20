import os
from typing import Dict, List, Optional, Tuple, Union
import json
from embedding import BaseEmbeddings
import numpy as np
from tqdm import tqdm
import faiss


class VectorStore:
    def __init__(self, document: List[str] = [""]) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 获得文档的向量表示
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = "storage"):
        # 数据库持久化，本地保存
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", "w", encoding="utf-8") as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding="utf-8") as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = "storage"):
        # 从本地加载数据库
        with open(f"{path}/vectors.json", "r", encoding="utf-8") as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", "r", encoding="utf-8") as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(
        self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1
    ) -> List[str]:
        # 根据问题检索相关的文档片段
        query_vector = EmbeddingModel.get_embedding(query)

        result = np.array(
            [self.get_similarity(query_vector, vector) for vector in self.vectors]
        )
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()


class FaissVetoreStore:
    def __init__(self, document: List[str] = [""]) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 获得文档的向量表示
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))

        self.index = faiss.IndexFlatL2(len(self.vectors[0]))  # 创建基于L2的索引
        return self.vectors

    def persist(self):
        # 数据库持久化，保存到faiss数据库
        if self.vectors:
            self.index.add(np.array(self.vectors).astype("float32"))

    def query(
        self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1
    ) -> List[str]:
        # 根据问题检索相关的文档片段
        query_vector = EmbeddingModel.get_embedding(query)
        query_vector = np.array([query_vector]).astype("float32")
        distance, index_result = self.index.search(query_vector, k)

        return np.array(self.document)[index_result.argsort()[-k:]].tolist()
