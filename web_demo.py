import gradio as gr
import time
from vector_base import VectorStore
from loader import ReadFiles
from model import InternLMChat, ZhipuChat
from embedding import JinaEmbedding, ZhipuEmbedding


def chat(message, history):
    vector = VectorStore()

    vector.load_vector("./storage")  # 加载本地的数据库

    embedding = ZhipuEmbedding()  # 创建EmbeddingModel

    content = vector.query(message, EmbeddingModel=embedding, k=1)[0]
    chat = ZhipuChat()
    yield chat.chat(message, [], content)


demo = gr.ChatInterface(
    fn=chat,
    examples=[
        "工作场所有害因素职业接触限值第1部分",
        "TinyRAG的项目结构是怎么样的",
    ],
    title="TinyRAG",
    theme="soft",
)
demo.launch(server_name="0.0.0.0", server_port=9001, share=True, inbrowser=True)
