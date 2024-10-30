# TinyRAG

A tiny RAG system.

- [TinyRAG](#tinyrag)
  - [设计](#设计)
  - [模块](#模块)
  - [项目结构](#项目结构)
  - [QuickStart](#quickstart)
  - [Todo](#todo)


项目参考 [QAnything](https://github.com/netease-youdao/qanything) 的设计并在 [KMnO4-zx/TinyRAG](https://github.com/KMnO4-zx/TinyRAG) 项目的基础之上进行了二次开发，增加了以下功能：

- 增加Faiss向量数据库支持
- 增加基于Gradio的web支持

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
    - 直接在history数据结构中记录过往的对话的rule和content

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

## Todo

- 对比不同embedding对结果的影响
- reranker和召回模型尽量不适用相同的base model
  - 多路召回：不同embedding数据库的向量召回 + 关键词搜索
  - 单路召回：便于数据管理，尤其是多模态场景
- 多轮对话
  - 直接将之前的对话结果传入模型会造成冗余，不精确
  - 尝试
    - 使用模型对历史会话做总结
    - 人为对query做意图理解
- 知识库构建
  - 细粒度知识要怎么保证简洁、精准
    - NER、成分句法分析等
  - 粗粒度知识怎么保证全面（跨行和跨段落的）
    - 多级检索方案
      - 一级索引关键信息：关键信息可以是段落的摘要的embedding
      - 二级索引原始文本
- 结果评估
  - 人工
