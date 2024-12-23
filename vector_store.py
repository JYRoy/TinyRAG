import os
from typing import Dict, List, Optional, Tuple, Union
import json
from embedding import BaseEmbeddings
import numpy as np
from tqdm import tqdm
import faiss

import jieba
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from loader import ReadFiles
import torch


class BM25(object):
    # 遍历文档，首先做分词，然后把分词后的文档和全文文档建立索引和映射关系
    def __init__(self, documents):
        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if len(line) < 5:
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))

        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    def _init_bm25(self):
        # 初始化BM25的知识库
        return BM25Retriever.from_documents(self.documents)

    def GetBM25TopK(self, query, topk):
        # 获得得分在topk的文档和分数
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans


class FaissRetriever(object):
    # 基于 langchain 中的 faiss 库的
    def __init__(self, model_path, data):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
        )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        del self.embeddings  # 使用完模型后释放显存
        torch.cuda.empty_cache()

    def GetTopK(self, query, k):
        # 获取top-K分数最高的文档块
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store


class VectorStore:
    # 基于 json 文件的
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
    # 直接基于 faiss 原生库的
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

    def update(self, EmbeddingModel: BaseEmbeddings, content: str):
        self.index.add(
            np.array([EmbeddingModel.get_embedding(content)]).astype("float32")
        )

    def query_history(
        self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1
    ) -> List[str]:
        # 根据问题检索相关的文档片段
        query_vector = EmbeddingModel.get_embedding(text=query)
        query_vector = np.array([query_vector]).astype("float32")
        distance, index_result = self.index.search(query_vector, k)
        return distance, index_result.argsort()[-k:]


if __name__ == "__main__":
    base = "."
    model_name = base + "/model/m3e-large"  # text2vec-large-chinese
    dp = ReadFiles(path=base + "/data/train_a.pdf")
    dp.parse_block(max_seq=1024)
    dp.parse_block(max_seq=512)
    print(len(dp.data))
    dp.parse_all_page(max_seq=256)
    dp.parse_all_page(max_seq=512)
    print(len(dp.data))
    dp.parse_one_page_with_rule(max_seq=256)
    dp.parse_one_page_with_rule(max_seq=512)
    print(len(dp.data))
    data = dp.data

    # faiss 召回
    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)

    # bm2.5 召回
    bm25 = BM25(data)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)
