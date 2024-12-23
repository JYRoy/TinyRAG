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
import pdfplumber
from PyPDF2 import PdfReader

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()
        self.data = []

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

    def sliding_window(self, sentences, kernel=512, stride=1):
        sz = len(sentences)
        cur = ""
        fast = 0
        slow = 0  # 慢指针指向窗口的起始句子
        while fast < len(sentences):
            sentence = sentences[fast]  # 快指针指向当前句子
            if len(cur + sentence) > kernel and (cur + sentence) not in self.data:
                # 如果合并之后会超过最大长度，并且当前内容不在 data 中，则合并在一起添加到 data 中
                self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。") :]  # 去掉slow指向的当前窗口的第一个句子，使用后面的内容来滑动窗口
                slow = slow + 1  # 慢指针向前移动，指向下一个句子
            # 如果还没有超过最大长度，继续合并
            cur = cur + sentence + "。"
            fast = fast + 1

    def data_filter(self, line, header, pageid, max_seq=1024):
        # 数据过滤，根据当前的文档内容的 item 划分句子，然后根据 max_seq 划分文档块。
        sz = len(line)
        if sz < 6:
            return

        if sz > max_seq:
            # 对于列表的情况，使用列表的方式进行划分
            if "■" in line:
                sentences = line.split("■")
            elif "•" in line:
                sentences = line.split("•")
            # 对于换行或者不同的子句子，使用换行符和句号进行划分
            elif "\t" in line:
                sentences = line.split("\t")
            else:
                sentences = line.split("。")

            # 按照换行拆成子句添加
            for subsentence in sentences:
                subsentence = subsentence.replace("\n", "")

                if len(subsentence) < max_seq and len(subsentence) > 5:
                    subsentence = (
                        subsentence.replace(",", "").replace("\n", "").replace("\t", "")
                    )
                    if subsentence not in self.data:
                        self.data.append(subsentence)
        else:
            # 没有超过最大长度，直接添加
            line = line.replace("\n", "").replace(",", "").replace("\t", "")
            if line not in self.data:
                self.data.append(line)

    def get_header(self, page):
        # 提取页头即一级标题
        try:
            lines = page.extract_words()[::]
        except:
            return None
        if len(lines) > 0:
            for line in lines:
                if "目录" in line["text"] or ".........." in line["text"]:
                    return None
                # if line["top"] < 20 and line["top"] > 17:  # 页头在 17-20 之间
                #     return line["text"]
            return lines[0]["text"]
        return None

    def parse_block(self, max_seq=1024):
        # 按照每页中块提取内容,并和一级标题进行组合,配合 Document 可进行意图识别
        # 尽可能保证一个小标题和对应的文档内容进行组合
        # 同时限定一个最大长度

        with pdfplumber.open(self._path) as pdf:

            for i, p in enumerate(pdf.pages):
                header = self.get_header(p)

                if header == None:
                    continue

                texts = p.extract_words(use_text_flow=True, extra_attrs=["size"])[::]

                squence = ""
                lastsize = 0

                for idx, line in enumerate(texts):
                    if idx < 1:  # 跳过 header
                        continue
                    if idx == 1:
                        if line["text"].isdigit():  # 跳过页脚的页码
                            continue

                    cursize = line["size"]  # 字体大小
                    text = line["text"]
                    if text == "□" or text == "•":
                        continue
                    elif (
                        text == "警告！" or text == "注意！" or text == "说明！"
                    ):  # 表示这是一整块内容，要放在一起
                        if len(squence) > 0:  # 判断上一行是否有内容
                            self.data_filter(squence, header, i, max_seq=max_seq)
                        squence = ""
                    elif format(lastsize, ".5f") == format(
                        cursize, ".5f"
                    ):  # 字体大小相同，说明是一个级别的内容，要合并到一起
                        if len(squence) > 0:
                            squence = squence + text
                        else:
                            squence = text
                    else:
                        # 当前内容是一页内容的开头
                        lastsize = cursize
                        if (
                            len(squence) < 15 and len(squence) > 0
                        ):  # 上一行是当前内容的小标题
                            squence = squence + text
                        else:
                            if len(squence) > 0:
                                self.data_filter(squence, header, i, max_seq=max_seq)
                            squence = text
                if len(squence) > 0:
                    self.data_filter(squence, header, i, max_seq=max_seq)

    def parse_one_page_with_rule(self, max_seq=512, min_len=6):
        # 按句号划分文档，然后利用最大长度划分文档块

        for idx, page in enumerate(PdfReader(self._path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if "...................." in text or "目录" in text:
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                page_content = page_content + text  # 将有内容的行拼接在一起
            if len(page_content) < min_len:
                continue
            if len(page_content) < max_seq:
                if page_content not in self.data:
                    self.data.append(page_content)
            else:
                sentences = page_content.split("。")  # 按照句子进行划分
                cur = ""
                for idx, sentence in enumerate(sentences):
                    # 如果没有达到最大长度，就一直拼接句子
                    # 到了最大程度，就添加到data中然后重新开始拼接后面的句子
                    if (
                        len(cur + sentence) > max_seq
                        and (cur + sentence) not in self.data
                    ):
                        self.data.append(cur + sentence)
                        cur = sentence
                    else:
                        cur = cur + sentence

    def parse_all_page(self, max_seq=512, min_len=6):
        #  滑窗法提取段落
        #  1. 把 pdf 看做一个整体,作为一个字符串
        #  2. 利用句号当做分隔符,切分成一个数组
        #  3. 利用滑窗法对数组进行滑动, 此处的
        # 作用：处理文本内容的跨页连续性问题
        all_content = ""
        # pdf 作为一个整体
        for idx, page in enumerate(PdfReader(self._path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if "...................." in text or "目录" in text:
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                page_content = page_content + text
            if len(page_content) < min_len:
                continue
            all_content = all_content + page_content
        # 然后按照句号进行切分
        sentences = all_content.split("。")
        # 在句子间进行滑动窗口
        self.sliding_window(sentences, kernel=max_seq)


if __name__ == "__main__":
    dp = ReadFiles(path="./data/train_a.pdf")
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
    out = open("all_text.txt", "w")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
