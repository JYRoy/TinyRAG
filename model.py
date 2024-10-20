import os
from typing import Dict, List, Optional, Tuple, Union

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
)


class BaseModel:
    def __init__(self, path: str = "") -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class ZhipuChat(BaseModel):
    def __init__(self, path: str = "", model: str = "glm-4-plus") -> None:
        super().__init__(path)
        from zhipuai import ZhipuAI

        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        history.append(
            {
                "role": "user",
                "content": PROMPT_TEMPLATE["RAG_PROMPT_TEMPALTE"].format(
                    question=prompt, context=content
                ),
            }
        )
        response = self.client.chat.completions.create(
            model=self.model, messages=history, max_tokens=150, temperature=0.1
        )
        return response.choices[0].message.content
