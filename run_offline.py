import time

from loader import ReadFiles
from vector_store import FaissRetriever, BM25

from vllm_model import ChatLLM

if __name__ == "__main__":

    start = time.time()

    base = "."
    qwen05 = base + "/model/Qwen-0.5B-Chat"
    m3e =  base + "/model/m3e-large"

    # 解析pdf文档，构造数据
    print("Reading PDF file...")
    dp = ReadFiles(path=base + "/data/train_a.pdf")
    # 三种提取方法整合在一起，尽可能地保证内容的完整性
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
    print("Reading PDF file finished.")
    
    # Faiss 召回
    faissretriever = FaissRetriever(m3e, data)
    vector_store = faissretriever.vector_store
    print("Faiss vector store finished.")
    
    # BM25召回
    bm25 = BM25(data)
    print("BM25 vector store finished.")
    
    llm = ChatLLM(qwen05)
    print("llm qwen load ok")
    
    
