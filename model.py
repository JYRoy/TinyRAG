import os
from typing import Dict, List, Optional, Tuple, Union
from vector_store import VectorStore, FaissVetoreStore
from embedding import BaseEmbeddings, BgeEmbedding
import torch
import config
from vllm import LLM, SamplingParams
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

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


# 获取stop token的id
def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(config.CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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
            history_response = self.search_history(self.history[-1]["content"])
            if history_response != None:
                return history_response["content"]

        self.save_history_to_faiss(self.history[-1])
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
        return response.choices[0].message.content

    def save_history_to_faiss(self, new_history: Dict):
        if new_history["role"] == "user":
            self.vector_store.update(self.embedding_model, new_history["content"])

    def search_history(self, query):
        distance, matched_query = self.vector_store.query_history(
            query=query, EmbeddingModel=self.embedding_model, k=1
        )

        if distance < 0.01:
            return self.all_history[matched_query[0][0] + 1]
        return None

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


class ChatLLM(object):

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token="<|extra_0|>",
            eos_token="<|endoftext|>",
            padding_side="left",
            trust_remote_code=True,
        )
        self.generation_config = GenerationConfig.from_pretrained(
            model_path, pad_token_id=self.tokenizer.pad_token_id
        )
        # self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        self.stop_words_ids = []

        # 加载vLLM大模型
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,  # 如果是多卡，可以自己把这个并行度设置为卡数N
            trust_remote_code=True,
            gpu_memory_utilization=0.6,  # 可以根据gpu的利用率自己调整这个比例
            dtype="bfloat16",
        )
        for stop_id in get_stop_words_ids(
            self.generation_config.chat_format, self.tokenizer
        ):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

        # LLM的采样参数
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "early_stopping": False,
            "top_p": 1.0,
            "top_k": (
                -1
                if self.generation_config.top_k == 0
                else self.generation_config.top_k
            ),
            "temperature": 0.0,
            "max_tokens": 2000,
            "repetition_penalty": self.generation_config.repetition_penalty,
            "n": 1,
            "best_of": 2,
            "use_beam_search": True,
        }
        self.sampling_params = SamplingParams(**sampling_kwargs)

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
        batch_text = []
        for q in prompts:
            raw_text, _ = make_context(
                self.tokenizer,
                q,
                system="You are a helpful assistant.",
                max_window_size=self.generation_config.max_window_size,
                chat_format=self.generation_config.chat_format,
            )
            batch_text.append(raw_text)
        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)
        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            if config.IMEND in output_str:
                output_str = output_str[: -len(config.IMEND)]
            if config.ENDOFTEXT in output_str:
                output_str = output_str[: -len(config.ENDOFTEXT)]
            batch_response.append(output_str)
        torch_gc()
        return batch_response


if __name__ == "__main__":
    base = "."
    qwen05 = base + "/model/Qwen-0.5B-Chat"
    start = time.time()
    llm = ChatLLM(qwen05)
    test = ["吉利汽车座椅按摩", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end - start) / 60))
