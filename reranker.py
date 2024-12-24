from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class reRankLLM(object):
    # 加载 rerank 模型
    def __init__(self, model_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.cuda()
        self.max_length = max_length

    def predict(self, query, docs):
        # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        response = [
            doc
            for score, doc in sorted(
                zip(scores, docs), reverse=True, key=lambda x: x[0]
            )
        ]
        torch_gc()
        return response


if __name__ == "__main__":
    bge_reranker_large = "./model/bge-reranker-large"
    bce_reranker_base = "./model/bce-reranker-base-v1"
    bge_rerank = reRankLLM(bge_reranker_large)
    bce_rerank = reRankLLM(bce_reranker_base)
