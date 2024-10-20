from vector_base import VectorStore
from loader import ReadFiles
from model import ZhipuChat
from embedding import ZhipuEmbedding, BgeEmbedding
from reranker import BgeReranker

import argparse


def none_vector_base(vector_path: str = "./storage"):
    # 没有保存数据库
    docs = ReadFiles("./data").get_content(
        max_token_len=600, cover_content=150
    )  # 获得data目录下的所有文件内容并分割
    vector = VectorStore(docs)
    embedding = BgeEmbedding()  # 创建EmbeddingModel
    reranker = BgeReranker()  # 创建RerankerModel
    vector.get_vector(EmbeddingModel=embedding)
    vector.persist(
        path=vector_path
    )  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

    question = "git的原理是什么？"

    content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
    rerank_content = reranker.rerank(question, content, k=2)
    best_content = rerank_content[0]
    chat = ZhipuChat()
    print(chat.chat(question, [], best_content))


def has_vector_base(vector_path: str = "./storage"):
    # 保存数据库之后
    vector = VectorStore()

    vector.load_vector(vector_path)  # 加载本地的数据库

    question = "git的原理是什么？"

    embedding = BgeEmbedding()  # 创建EmbeddingModel

    reranker = BgeReranker()  # 创建RerankerModel

    content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
    rerank_content = reranker.rerank(question, content, k=2)
    best_content = rerank_content[0]
    chat = ZhipuChat()
    print(chat.chat(question, [], content))


# if __name__ == "__main__":
#     # 创建解析步骤
#     parser = argparse.ArgumentParser(description="chat")

#     # 添加参数步骤
#     parser.add_argument("vector_base", metavar="V", type=str, help="vector base path")

#     args = parser.parse_args()

#     if args.vector_base != None:
#         none_vector_base(args.vector_base)
#     else:
#         has_vector_base(args.vector_base)

none_vector_base()
has_vector_base()
