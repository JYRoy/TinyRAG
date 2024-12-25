import os
import torch
from config import *
from vllm import LLM, SamplingParams
import time
from transformers import AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatLLM(object):

    def __init__(self, model_path):
        self.model_path = model_path
        if "qwen" in self.model_path or "Qwen" in self.model_path:
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
            self.tokenizer.eos_token_id = self.generation_config.eos_token_id
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
        elif "chatglm" in self.model_path or "ChatGLM" in self.model_path:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "./model/ChatGLM3-6B", trust_remote_code=True
            )
            self._model = LLM(
                model="./model/ChatGLM3-6B",
                tensor_parallel_size=2,
                trust_remote_code=True,
                max_model_len=1024,
                gpu_memory_utilization=0.8,
                enforce_eager=True,
                dtype="bfloat16",
            )
            stop_token_ids = [151329, 151336, 151338]
            self._sampling_params = SamplingParams(
                temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids
            )
        elif "baichuan" in self.model_path or "Baichuan" in self.model_path:
            self._model = LLM(
                model="./model/Baichuan2-7B-Chat",
                tensor_parallel_size=2,
                trust_remote_code=True,
                gpu_memory_utilization=0.8,
                dtype="bfloat16",
            )
        else:
            raise NotImplementedError(f"Unknown model {self.model_path!r}")

    def infer(self, prompts):
        if "qwen" in self.model_path or "Qwen" in self.model_path:
            return self.infer_qwen(prompts)
        elif "chatglm" in self.model_path or "ChatGLM" in self.model_path:
            return self.infer_chatglm(prompts)
        elif "baichuan" in self.model_path or "Baichuan" in self.model_path:
            return self.infer_baichuan(prompts)
        else:
            raise NotImplementedError(f"Unknown model {self.model_path!r}")

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer_qwen(self, prompts):
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
            if IMEND in output_str:
                output_str = output_str[: -len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[: -len(ENDOFTEXT)]
            batch_response.append(output_str)
        torch_gc()
        return batch_response

    def infer_baichuan(self, prompts):
        output = self._model.generate(
            prompts, sampling_params=SamplingParams(temperature=0.95, max_tokens=1024)
        )
        torch_gc()
        return output[0].outputs[0].text

    def infer_chatglm(self, prompts):
        output = self._model.generate(
            prompts=prompts, sampling_params=self._sampling_params
        )
        torch_gc()
        return output[0].outputs[0].text


if __name__ == "__main__":
    base = "."
    qwen_7b = base + "/model/Qwen-7B-Chat"
    chatglm3_6b = base + "/model/ChatGLM3-6B"
    baichuan2_7b = base + "/model/Baichuan2-7B-Chat"
    test = ["吉利汽车座椅按摩", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]

    start = time.time()
    llm_qwen_7b = ChatLLM(qwen_7b)
    qwen_7b_generated_text = llm_qwen_7b.infer(test)
    print(qwen_7b_generated_text)
    end = time.time()
    print("qwen7b cost time: " + str((end - start) / 60))

    start = time.time()
    llm_chatglm3_6b = ChatLLM(chatglm3_6b)
    chatglm3_6b_generated_text = llm_chatglm3_6b.infer(test)
    print(chatglm3_6b_generated_text)
    end = time.time()
    print("chatglm3_6b cost time: " + str((end - start) / 60))

    start = time.time()
    llm_baichuan2_7b = ChatLLM(baichuan2_7b)
    baichuan2_7b_generated_text = llm_baichuan2_7b.infer(test)
    print(baichuan2_7b_generated_text)
    end = time.time()
    print("baichuan2_7b cost time: " + str((end - start) / 60))
