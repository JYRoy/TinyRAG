import os

import PyPDF2
import tiktoken
import fitz
import spacy
from apryse_sdk import PDFNet, HTMLOutputOptions, Convert
import pickle
from pathlib import Path
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.node_parser import SimpleNodeParser

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".pdf"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_pdf_by_llama_parse(file)
            chunk_content = self.get_chunk_by_spacy(
                content, max_token_len=max_token_len, cover_content=cover_content
            )
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ""

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # 假设以换行符\n分割文本为行

        for line in lines:
            line = line.replace(" ", "")
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # 避免跨单词分割
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                # 处理最后一个块
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                chunk_text.append(curr_chunk)

            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += "\n"
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)
        return chunk_text

    @classmethod
    def get_chunk_by_spacy(
        cls, text: str, max_token_len: int = 600, cover_content: int = 150
    ):
        chunk_text = []

        nlp = spacy.load("zh_core_web_sm")
        doc = nlp(text)
        for s in doc.sents:
            chunk_text.append(s.text)
        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith(".pdf"):
            return cls.read_pdf(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_pdf_by_pymupdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, "rb") as file:
            doc = fitz.open(file)
            text = ""
            num_pages = doc.page_count
            for page_index in range(num_pages):
                page = doc.load_page(page_index)
                text += page.get_text()
            return text

    @classmethod
    def read_pdf_by_llama_parse(cls, file_path: str):
        from llama_parse import LlamaParse

        # 创建一个LlamaParse对象，传入OpenAIAPIKey和注册后获得的LlamaParseAPIKey。
        parser_gpt = LlamaParse(
            result_type="markdown",
            api_key="llx-TqVzvRvIPVrplEHnJ4BxSzp1rD5vfPtIBQQSOv5cyZvk9VAz",
        )
        pdf_file = "data/demo.pdf"
        pkl_file = "output/demo.pkl"
        if not os.path.exists(pkl_file):
            # 将PDF文件转换为Markdown格式内容
            documents_gpt = parser_gpt.load_data(pdf_file)
            # 转换后的Markdown内容将保存在demo. pkl文件中
            pickle.dump(documents_gpt, open(pkl_file, "wb"))
        else:
            # 将转换后的Markdown内容保存到documents_gpt变量中
            documents_gpt = pickle.load(open(pkl_file, "rb"))
        return_text = []
        for i in range(len(documents_gpt)):
            return_text.append(documents_gpt[i].text)
        return "".join(return_text)
