# TinyRAG

A tiny RAG system.

- [TinyRAG](#tinyrag)
  - [设计](#设计)
  - [模块](#模块)
  - [项目结构](#项目结构)
  - [QuickStart](#quickstart)
  - [TODO](#todo)


项目参考 [QAnything](https://github.com/netease-youdao/qanything) 的设计并在 [KMnO4-zx/TinyRAG](https://github.com/KMnO4-zx/TinyRAG) 项目的基础之上进行了二次开发，增加了以下功能：

- 增加Faiss向量数据库支持
- 增加基于Gradio的web支持
- 基于gpt和spacy的pdf文档分块

并对部分功能进行删减：

- Embedding模块只支持BGEEmbedding和ZHIPU API方式的embedding
- 模型模块只支持ZHIPU的glm-4-plus模型的对话

## 设计

![architecture](./images/architecture.png)

## 模块

- 文档加载和切分模块：用来加载文档并切分成文档片段，支持pdf文件
- 向量化模块：用来将切分后的文档片段向量化，使用bge模型
- 数据库：存放文档片段和对应的向量，使用faiss向量数据库
- 检索（召回）模块：实现单路召回用来根据 Query （问题）检索相关的文档片段，在faiss数据库中召回top-n
- 重排模块：使用检索（召回）模块的结果，使用bge重排模型进行重排
- 模型模块：用来根据检索出来的文档和用户的输入，回答用户的问题
  - 多轮对话
    - 支持全量历史对话和长短时记忆历史对话
    - 支持聊天记录的向量数据库存储

## 项目结构

- loader.py
  - ReadFiles：用于读取并分割pdf文件，实现了两种解析pdf的方式
    - 第一种：使用PyPDF2库直接对文本解析，缺点是无法很好的处理表格和不同级别的内容；
    - 第二种：使用llama_parser库将pdf文件转换为markdown文件，优点是可以保留不同标题级别，解析出表格和图片（是使用多模态模型），缺点是准确率比较低，且llama_parser库依赖OpenAI访问；
  - get_chunk：将内容分块，实现了两种分块方式
    - 第一种：递归字符切分方式，设置一个chunk_size作为当前窗口长度和一个cover_content作为重叠内容长度，以行为单位进行切分，对于一行内容长度超过chunk_size + cover_content长度的，切分为多个块，否则按行保留内容。有点是可以限制分块长度便于管理，缺点是无法保证上下文的连贯性（cover_content就是为了尽可能减小这部分的损失的），尤其是跨行和跨段落的内容；
    - 第二种：使用spacy库，直接切分，因为spacy库中的模型都是已经训练过的，所以基于spacy的优点是可以根据上下文进行更高质量的切分；
- embedding.py
  - BaseEmeddings
  - BgeEmbedding：使用BGE模型，获取字符串对应的embedding向量
  - ZhipuEmbedding：通过zhipu api获取字符串对应的embedding向量
- vector_store.py
  - VectorStore：采用本地json存储分割后的文档以及embedding的方式
  - FaissVectorStore：使用Faiss数据库存储embedding的方式
- reranker.py
  - BgeReranker：使用BEG ReRanker模型对召回的结果进行重排
- model.py
  - ZhipuChat：支持使用zhipu glm-4-plus模型进行多轮对话
- local_demo.py
  - 作为基于json和基于faiss数据库对话的测试文件
- web_demo.py
  - 使用Gradio可视化

## QuickStart

配置 huggingface 镜像代理

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

安装依赖包

```shell
pip install -r requirements.txt
```

如果使用 spacy，需要执行命令来下载

```shell
python3 -m spacy download zh_core_web_sm
```

设置ZHIPU API KEY（需要去 [zhipu bigmodel](https://open.bigmodel.cn/usercenter/apikeys) 申请）

```shell
export ZHIPUAI_API_KEY=xxxxxxxxx
```

启动web

```shell
python3 web_demo.py
```

浏览器访问：

```shell
http://localhost:9001/
```

## TODO

- 知识库构建
  - 细粒度知识要怎么保证简洁、精准（现实中不太会这么用）
    - NER、成分句法分析等，但是可能有准确度问题，表格的格式版式可能不固定
  - 粗粒度知识怎么保证全面（跨行和跨段落的）
    - 多级检索方案（1. 额外信息引入可能效果 2. 性能不一定更好，系统设计更复杂，向量数据库的检索性能本来就很快）
      - 一级索引关键信息：关键信息可以是段落的摘要的embedding
      - 二级索引原始文本
    - 基于 bert 的 NSP 方案：如果是连贯的，作为一句话存储
  - PDF 版面分析
  - 多种文件格式支持
    - html格式/markdown：
      - 规则匹配：可能有文字长度的问题
    - 不同文件格式可以统一
  - 多模态支持
    - 1. 原始图片转为embedding
    - 2. 使用多模态模型获取图片的summary保存为embedding
  - OCR + 大模型
  - 通用方案：多模态大模型 + 特定场景特定处理
  - GraphRAG项目或图建立知识库
- Embedding
  - 对比不同embedding对结果的影响
- 召回、ReRank
  - reranker和召回模型尽量不适用相同的base model
  - 多路召回：不同embedding数据库的向量召回 + 关键词搜索
    - 策略互补：字面相似 + 向量相似性
    - 加权融合分数、取各自topk检索后并集或者RRF+Rerank
  - 单路召回：便于数据管理，尤其是多模态场景
- 多轮对话
  - 直接将之前的对话结果传入模型会造成冗余，不精确
  - 不同的策略适用于不同的场景
    - 使用模型对历史会话做总结
      - 历史所有对话做整体总结
      - 长短时记忆：滑动窗口 + 时间衰减方式，对较早的对话做总结，较新的对话保留原文
      - 总结用户问题和模型的回答结果作为一条信息
      - summary的模型通常比较小，需要根据模型情况设计history的窗口大小，保证输出质量
    - 构建历史对话向量数据库
      - 1.可以把相近问题的答案直接抛出来
      - 2.提取关键的历史信息
    - 滑动窗口方式
    - 关键词提取
      - 实体提取：NER提取历史对话中的实体信息
      - 流程：新query来了之后，先查上文的几条历史信息的关键词，将提取出来的关键词和当前的query组合起来进行语义改写，用改写后的query来查询
    - 人为对query做意图理解
- 结果评估
  - 人工


长短时记忆：

Naive方式：

histoy = [
  {"role":sys, content: ""},
  {"role":user, content: ""},
  {"role":sys, content: ""},
  {"role":user, content: ""},
]

current_query = {"role":user, content: ""},

长短记忆方式：

histoy = [
  {"role":sys, content: "用户问了XXX，模型回答是XXX"},
  current_query = {"role":user, content: ""},
]
