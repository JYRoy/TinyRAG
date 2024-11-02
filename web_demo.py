import gradio as gr
import time
from vector_store import VectorStore, FaissVetoreStore
from loader import ReadFiles
from model import ZhipuChat
from embedding import ZhipuEmbedding, BgeEmbedding
from reranker import BgeReranker


chat_model = ZhipuChat()


def chat(message, history):
    docs = ReadFiles("./data").get_content(
        max_token_len=600, cover_content=150
    )  # 获得data目录下的所有文件内容并分割
    vector = FaissVetoreStore(docs)

    embedding = BgeEmbedding()  # 创建EmbeddingModel
    reranker = BgeReranker()  # 创建RerankerModel
    vector.get_vector(EmbeddingModel=embedding)
    vector.persist()

    content = vector.query(message, EmbeddingModel=embedding, k=10)
    rerank_content = reranker.rerank(message, content[0], k=10)
    best_content = rerank_content[0]

    yield chat_model.chat(message, best_content)


demo = gr.ChatInterface(
    fn=chat,
    examples=[
        "工作场所有害因素职业接触限值第1部分",
        "TinyRAG的项目结构是怎么样的",
    ],
    title="TinyRAG",
    theme="soft",
)
demo.launch(server_name="localhost", server_port=9002, share=True, inbrowser=True)
