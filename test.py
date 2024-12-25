import time
import json
from loader import ReadFiles
from retriever import FaissRetriever, BM25, TFIDF
from reranker import reRankLLM
from model import ChatLLM


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


def get_emb_bm25_merge(faiss_context, bm25_context, query):
    # 构造提示，根据 merged faiss 和 bm25 的召回结果返回答案
    max_length = 2500
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans = bm25_ans + doc.page_content
        if cnt > 6:
            break

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                2: {bm25_ans}
                                问题:
                                {question}""".format(
        emb_ans=emb_ans, bm25_ans=bm25_ans, question=query
    )
    return prompt_template


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


def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    emb_ans = ""
    for doc in rerank_ans:
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans


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


if __name__ == "__main__":

    start = time.time()

    base = "."
    qwen7b = base + "/model/Qwen-7B-Chat"
    m3e = base + "/model/m3e-large"
    bge = base + "/model/bge-large-zh"
    gte = base + "/model/gte-base-zh"
    bce = base + "/model/bce-base-v1"
    bge_reranker_large = base + "/model/bge-reranker-large"
    bce_reranker_base = base + "/model/bce-reranker-base-v1"

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
    bge_rerank = reRankLLM(bge_reranker_large)
    bce_rerank = reRankLLM(bce_reranker_base)
    print("reRank model load ok.")

    # 对每一条测试问题，做答案生成处理
    with open(base + "/data/test_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]
            # m3e faiss 召回 topk
            m3e_faiss_context = m3e_faissretriever.GetTopK(query, 15)
            m3e_faiss_min_score = 0.0
            if len(m3e_faiss_context) > 0:
                m3e_faiss_min_score = m3e_faiss_context[0][1]
            cnt = 0
            m3e_emb_ans = ""
            for doc, score in m3e_faiss_context:
                cnt = cnt + 1
                # 最长选择 max length
                if len(m3e_emb_ans + doc.page_content) > max_length:
                    break
                m3e_emb_ans = m3e_emb_ans + doc.page_content
                # 最多选择6个
                if cnt > 6:
                    break

            # bge faiss 召回 topk
            bge_faiss_context = bge_faissretriever.GetTopK(query, 15)
            bge_faiss_min_score = 0.0
            if len(bge_faiss_context) > 0:
                bge_faiss_min_score = bge_faiss_context[0][1]
            cnt = 0
            bge_emb_ans = ""
            for doc, score in bge_faiss_context:
                cnt = cnt + 1
                # 最长选择 max length
                if len(bge_emb_ans + doc.page_content) > max_length:
                    break
                bge_emb_ans = bge_emb_ans + doc.page_content
                # 最多选择6个
                if cnt > 6:
                    break

            # gte faiss 召回 topk
            gte_faiss_context = gte_faissretriever.GetTopK(query, 15)
            gte_faiss_min_score = 0.0
            if len(gte_faiss_context) > 0:
                gte_faiss_min_score = gte_faiss_context[0][1]
            cnt = 0
            gte_emb_ans = ""
            for doc, score in gte_faiss_context:
                cnt = cnt + 1
                # 最长选择 max length
                if len(gte_emb_ans + doc.page_content) > max_length:
                    break
                gte_emb_ans = gte_emb_ans + doc.page_content
                # 最多选择6个
                if cnt > 6:
                    break

            # bce faiss 召回 topk
            bce_faiss_context = bce_faissretriever.GetTopK(query, 15)
            bce_faiss_min_score = 0.0
            if len(bce_faiss_context) > 0:
                bce_faiss_min_score = bce_faiss_context[0][1]
            cnt = 0
            bce_emb_ans = ""
            for doc, score in bce_faiss_context:
                cnt = cnt + 1
                # 最长选择 max length
                if len(bce_emb_ans + doc.page_content) > max_length:
                    break
                bce_emb_ans = bce_emb_ans + doc.page_content
                # 最多选择6个
                if cnt > 6:
                    break

            # bm2.5 召回 topk
            bm25_context = bm25.GetBM25TopK(query, 15)
            bm25_ans = ""
            cnt = 0
            for doc in bm25_context:
                cnt = cnt + 1
                if len(bm25_ans + doc.page_content) > max_length:
                    break
                bm25_ans = bm25_ans + doc.page_content
                if cnt > 6:
                    break

            # tfidf 召回 topk
            tfidf_context = tfidf.GetTFIDFTopK(query, 15)
            tfidf_ans = ""
            cnt = 0
            for doc in tfidf_context:
                cnt = cnt + 1
                if len(tfidf_ans + doc.page_content) > max_length:
                    break
                tfidf_ans = tfidf_ans + doc.page_content
                if cnt > 6:
                    break

            # 构造合并 bm25 召回和 m3e 向量召回的 prompt
            emb_bm25_merge_inputs = get_emb_bm25_merge(
                m3e_faiss_context, bm25_context, query
            )

            # 构造合并所有路召回的 prompt
            emb_all_merge_inputs = get_emb_all_merge(
                m3e_faiss_context,
                bge_faiss_context,
                gte_faiss_context,
                bce_faiss_context,
                bm25_context,
                tfidf_context,
                query,
            )

            # 构造bm25召回的prompt
            bm25_inputs = get_rerank(bm25_ans, query)

            # 构造向量召回的prompt
            emb_inputs = get_rerank(m3e_emb_ans, query)

            # 使用 bge rerank m3e 和 bm25 召回的候选，并按照相关性得分排序
            rerank_m3e_bm25_ans = reRank(
                bge_rerank, 6, query, bm25_context, m3e_faiss_context
            )

            # 使用 bge rerank 多路召回的候选，并按照相关性得分排序
            bge_rerank_all_ans = reRankAll(
                bge_rerank,
                6,
                query,
                m3e_faiss_context,
                bge_faiss_context,
                gte_faiss_context,
                bce_faiss_context,
                bm25_context,
                tfidf_context,
            )

            # 使用 bge rerank 多路召回的候选，并按照相关性得分排序
            bce_rerank_all_ans = reRankAll(
                bce_rerank,
                6,
                query,
                m3e_faiss_context,
                bge_faiss_context,
                gte_faiss_context,
                bce_faiss_context,
                bm25_context,
                tfidf_context,
            )

            # 构造得到 rerank 后生成答案的prompt
            rerank_m3e_bm25_inputs = get_rerank(rerank_m3e_bm25_ans, query)

            # 构造得到 bge_rerank_all_ans 后生成答案的prompt
            bge_rerank_all_inputs = get_rerank(bge_rerank_all_ans, query)

            # 构造得到 bge_rerank_all_ans 后生成答案的prompt
            bce_rerank_all_inputs = get_rerank(bce_rerank_all_ans, query)

            batch_input = []
            batch_input.append(emb_all_merge_inputs)
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(emb_inputs)
            batch_input.append(rerank_m3e_bm25_inputs)
            batch_input.append(bge_rerank_all_inputs)
            batch_input.append(bce_rerank_all_inputs)
            # 执行 batch 推理
            batch_output = llm.infer(batch_input)
            line["answer_0"] = batch_output[0]  # 合并多路召回的结果
            line["answer_1"] = batch_output[1]  # 合并两路召回的结果
            line["answer_2"] = batch_output[2]  # bm召回的结果
            line["answer_3"] = batch_output[3]  # 向量召回的结果
            line["answer_4"] = batch_output[4]  # 两路召回 bge 重排序后的结果
            line["answer_5"] = batch_output[5]  # 多路召回 bge 重排序后的结果
            line["answer_6"] = batch_output[6]  # 多路召回 bce 重排序后的结果
            line["answer_7"] = m3e_emb_ans
            line["answer_8"] = bm25_ans
            line["answer_9"] = rerank_m3e_bm25_ans
            line["answer_10"] = bge_rerank_all_ans
            # 如果 faiss 检索跟 query 的距离高于 500，输出无答案
            if m3e_faiss_min_score > 500:
                line["answer_7"] = "无答案"
            else:
                line["answer_7"] = str(m3e_faiss_min_score)

        # 保存结果，生成 submission 文件
        json.dump(
            jdata,
            open(base + "/data/result.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        end = time.time()
        print("cost time: " + str(int(end - start) / 60))
