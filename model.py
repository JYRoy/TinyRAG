import os
from typing import Dict, List, Optional, Tuple, Union
from vector_store import VectorStore, FaissVetoreStore
import faiss
from embedding import BaseEmbeddings, BgeEmbedding

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    HISTORY_TEMPLATE="""
        请对给出的对话历史进行总结，对话历史为：
        ···
        {history}
        ···
    """,
)


class BaseModel:
    def __init__(self, path: str = "") -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class ZhipuChat(BaseModel):
    def __init__(
        self,
        path: str = "",
        model: str = "glm-4-plus",
        embedding_model: BaseEmbeddings = BgeEmbedding,
    ) -> None:
        super().__init__(path)
        from zhipuai import ZhipuAI

        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.model = model
        self.history_window = 2  # 必须为偶数值
        self.history: List[Dict] = []
        self.vector_store = FaissVetoreStore()
        self.embedding_model = embedding_model()
        self.vector_store.get_vector(self.embedding_model)
        self.all_history = []

    def chat(self, prompt: str, content: str) -> str:
        self.history.append(
            {
                "role": "user",
                "content": PROMPT_TEMPLATE["RAG_PROMPT_TEMPALTE"].format(
                    question=prompt, context=content
                ),
            }
        )
        self.all_history.append(self.history[-1])
        if len(self.history) > 1:
            self.search_history(self.history[-1]["content"])

        self.history = self.sum_history()

        response = self.client.chat.completions.create(
            model=self.model, messages=self.history, max_tokens=150, temperature=0.1
        )
        self.history.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        )
        self.all_history.append(self.history[-1])
        self.save_history_to_faiss(self.history[-1])
        return response.choices[0].message.content

    def save_history_to_faiss(self, new_history: Dict):
        if new_history["role"] == "assistant":
            self.vector_store.update(self.embedding_model, new_history["content"])

    def search_history(self, query):
        matched_query = self.vector_store.query_history(
            query=query, EmbeddingModel=self.embedding_model, k=1
        )
        import pdb; pdb.set_trace()
        return self.all_history[matched_query]

    def sum_history(self):
        if len(self.history) - 1 > self.history_window:
            lens = len(self.history)
            summary_str = self.history[0 : lens - self.history_window + 1]
            formatted_strings = [
                ", ".join(f"{key}, {value}" for key, value in d.items())
                for d in summary_str
            ]
            result_string = "，".join(formatted_strings)
            history_message = [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE["HISTORY_TEMPLATE"].format(
                        history=result_string
                    ),
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=history_message,
                max_tokens=300,
                temperature=0.1,
            )

            summary = [
                {"role": "assistant", "content": response.choices[0].message.content}
            ]

            summary.extend(self.history[lens - self.history_window + 1 :])

            return summary
        else:
            return self.history
