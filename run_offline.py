import json
from argparse import ArgumentParser

from loader import ReadFiles
from retriever import FaissRetriever, BM25, TFIDF
from reranker import reRankLLM
from model import ChatLLM


def arg_parse():

    parser = ArgumentParser()
    parser.add_argument(
        "--llm_model_path", default="model/Qwen-7B-Chat", help="the path of llm to use"
    )
    parser.add_argument(
        "--rerank_model_path",
        default="model/bge-reranker-large",
        help="the path of rerank model to use",
    )
    parser.add_argument("--data_path", default="data/train_a.pdf", help="data path")
    parser.add_argument(
        "--test_query_path", default="data/test_question.json", help="test query path"
    )

    return parser.parse_args()


def get_rerank(emb_ans, query):

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(
        emb_ans=emb_ans, question=query
    )
    return prompt_template


def document_polish(documents):
    documents_prompt = """你是一个友善的助手，可以将输入的内容重写为更易理解的表述。输入是：{documents}""".format(
        documents=documents
    )
    return documents_prompt


def question_polish(question):
    new_question = """你是一个友善的助手，可以将输入的问题扩写为更加详细和合适的问题。问题是：{question}""".format(
        question=question
    )
    return question


def answer_polish(question, documents, answer):
    new_answer = """
        你是一个友善的助手，对输入的问题根据输入背景，润色已回答的内容，达到更好的质量。\n
        如果背景和问题无关，请回答“无关”，不允许在回答中添加编造成分，回答请使用中文。\n
        问题是：{question},        
        背景是：{documents},
        回答是：{answer},
        润色结果为：
        """.format(
        question=question, documents=documents, answer=answer
    )
    return new_answer


def reRankAll(
    rerank,
    top_k,
    query,
    m3e_faiss_ans,
    bge_faiss_ans,
    gte_faiss_ans,
    bce_faiss_ans,
    bm25_ans,
    tfidf_ans,
):

    items = []
    max_length = 4000
    for doc, score in m3e_faiss_ans:
        items.append(doc)
    for doc, score in bge_faiss_ans:
        items.append(doc)
    for doc, score in gte_faiss_ans:
        items.append(doc)
    for doc, score in bce_faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    items.extend(tfidf_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    emb_ans = ""
    for doc in rerank_ans:
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans


def get_emb_all_merge(
    m3e_faiss_context,
    bge_faiss_context,
    gte_faiss_context,
    bce_faiss_context,
    bm25_context,
    tfidf_context,
    query,
):
    # 构造提示，根据输入的多路召回结果返回答案
    max_length = 2500
    m3e_emb_ans = ""
    cnt = 0
    for doc, score in m3e_faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(m3e_emb_ans + doc.page_content) > max_length:
            break
        m3e_emb_ans = m3e_emb_ans + doc.page_content

    bge_emb_ans = ""
    cnt = 0
    for doc, score in bge_faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(bge_emb_ans + doc.page_content) > max_length:
            break
        bge_emb_ans = bge_emb_ans + doc.page_content

    gte_emb_ans = ""
    cnt = 0
    for doc, score in gte_faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(gte_emb_ans + doc.page_content) > max_length:
            break
        gte_emb_ans = gte_emb_ans + doc.page_content

    bce_emb_ans = ""
    cnt = 0
    for doc, score in bce_faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(bce_emb_ans + doc.page_content) > max_length:
            break
        bce_emb_ans = bce_emb_ans + doc.page_content

    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans = bm25_ans + doc.page_content
        if cnt > 6:
            break

    tfidf_ans = ""
    cnt = 0
    for doc in tfidf_context:
        cnt = cnt + 1
        if len(tfidf_ans + doc.page_content) > max_length:
            break
        tfidf_ans = tfidf_ans + doc.page_content
        if cnt > 6:
            break

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {m3e_emb_ans}
                                2: {bge_emb_ans}
                                3: {gte_emb_ans}
                                4: {bce_emb_ans}
                                5: {bm25_ans}
                                6: {tfidf_ans}
                                问题:
                                {question}""".format(
        m3e_emb_ans=m3e_emb_ans,
        bge_emb_ans=bge_emb_ans,
        gte_emb_ans=gte_emb_ans,
        bce_emb_ans=bce_emb_ans,
        bm25_ans=bm25_ans,
        tfidf_ans=tfidf_ans,
        question=query,
    )
    return prompt_template


def main():
    args = arg_parse()

    dp = ReadFiles(path=args.data_path)
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

    base = "."
    qwen7b = base + "/model/Qwen-7B-Chat"
    m3e = base + "/model/m3e-large"
    bge = base + "/model/bge-large-zh"
    gte = base + "/model/gte-base-zh"
    bce = base + "/model/bce-base-v1"

    # Faiss 召回: M3E
    m3e_faissretriever = FaissRetriever(m3e, data)
    print("M3E Faiss vector store finished.")

    # faiss 召回: bge-large
    bge_faissretriever = FaissRetriever(bge, data)
    print("BGE Faiss vector store finished.")

    # faiss 召回: gte-base
    gte_faissretriever = FaissRetriever(gte, data)
    print("GTE Faiss vector store finished.")

    # faiss 召回: bce-base
    bce_faissretriever = FaissRetriever(bce, data)
    print("BCE Faiss vector store finished.")

    # BM25召回
    bm25 = BM25(data)
    print("BM25 vector store finished.")

    # TFIDF 召回
    tfidf = TFIDF(data)
    print("TFIDF vector store finished.")

    llm = ChatLLM(qwen7b)
    print("Qwen model load ok.")

    # reRank模型
    rerank = reRankLLM(args.rerank_model_path)
    print("reRank model load ok.")

    # 对每一条测试问题，做答案生成处理
    with open(args.test_query_path, "r") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]
            # 直接推理生成的答案
            ans = llm.infer([query])

            # 使用 llm 对问题进行扩写，用于提升检索效果
            query = question_polish(query)
            new_query = llm.infer([query])
            # 将扩写的问题和直接推理生成的答案拼接，用于提升检索效果
            new_query = query + new_query[0] + ans[0]

            # m3e faiss 召回 topk
            m3e_faiss_context = m3e_faissretriever.GetTopK(new_query, 6)
            m3e_faiss_min_score = 0.0
            if len(m3e_faiss_context) > 0:
                m3e_faiss_min_score = m3e_faiss_context[0][1]
            m3e_emb_ans = ""
            for doc, score in m3e_faiss_context:
                # 最长选择 max length
                if len(m3e_emb_ans + doc.page_content) > max_length:
                    break
                m3e_emb_ans = m3e_emb_ans + doc.page_content

            # bge faiss 召回 topk
            bge_faiss_context = bge_faissretriever.GetTopK(new_query, 6)
            bge_faiss_min_score = 0.0
            if len(bge_faiss_context) > 0:
                bge_faiss_min_score = bge_faiss_context[0][1]
            bge_emb_ans = ""
            for doc, score in bge_faiss_context:
                # 最长选择 max length
                if len(bge_emb_ans + doc.page_content) > max_length:
                    break
                bge_emb_ans = bge_emb_ans + doc.page_content

            # gte faiss 召回 topk
            gte_faiss_context = gte_faissretriever.GetTopK(new_query, 6)
            gte_faiss_min_score = 0.0
            if len(gte_faiss_context) > 0:
                gte_faiss_min_score = gte_faiss_context[0][1]
            gte_emb_ans = ""
            for doc, score in gte_faiss_context:
                # 最长选择 max length
                if len(gte_emb_ans + doc.page_content) > max_length:
                    break
                gte_emb_ans = gte_emb_ans + doc.page_content

            # bce faiss 召回 topk
            bce_faiss_context = bce_faissretriever.GetTopK(new_query, 6)
            bce_faiss_min_score = 0.0
            if len(bce_faiss_context) > 0:
                bce_faiss_min_score = bce_faiss_context[0][1]
            bce_emb_ans = ""
            for doc, score in bce_faiss_context:
                # 最长选择 max length
                if len(bce_emb_ans + doc.page_content) > max_length:
                    break
                bce_emb_ans = bce_emb_ans + doc.page_content

            # bm2.5 召回 topk
            bm25_context = bm25.GetBM25TopK(new_query, 6)
            bm25_ans = ""
            for doc in bm25_context:
                if len(bm25_ans + doc.page_content) > max_length:
                    break
                bm25_ans = bm25_ans + doc.page_content

            # tfidf 召回 topk
            tfidf_context = tfidf.GetBM25TopK(new_query, 6)
            tfidf_ans = ""
            for doc in tfidf_context:
                if len(tfidf_ans + doc.page_content) > max_length:
                    break
                tfidf_ans = tfidf_ans + doc.page_content

            # 使用 rerank 多路召回的候选，并按照相关性得分排序
            rerank_all_ans = reRankAll(
                rerank,
                6,
                query,
                m3e_faiss_context,
                bge_faiss_context,
                gte_faiss_context,
                bce_faiss_context,
                bm25_context,
                tfidf_context,
            )

            # 对提取出来的文档内容进行润色，后送入 llm 中推理
            rerank_all_ans = document_polish(rerank_all_ans)
            rerank_all_ans = llm.infer([rerank_all_ans])

            # 构造得到使用润色和扩写后的 rerank 结果和query 生成答案的 prompt
            rerank_all_inputs = get_rerank(rerank_all_ans[0], query)

            # 执行 batch 推理
            batch_output = llm.infer([rerank_all_inputs])
            line["answer_0"] = batch_output[0]  # 多路召回重排序后的结果
            # 如果 faiss 检索跟 query 的距离高于 500，输出无答案
            faiss_min_score = min(
                m3e_faiss_min_score,
                bge_faiss_min_score,
                gte_faiss_min_score,
                bce_faiss_min_score,
            )
            if faiss_min_score > 500:
                line["answer_0"] = "无答案"

            # 对生成的答案进行一次润色
            new_answer = answer_polish(query, rerank_all_ans[0], line["answer_0"])
            new_answer = llm([new_answer])
            line["answer_0"] = new_answer[0]

        # 保存结果，生成 submission 文件
        json.dump(
            jdata,
            open(base + "/data/result.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
