"""
文本分块器 - 将长文档分割成较小的块
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


class TextSplitter:
    """文本分块器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # 段落分隔
                "\n",    # 换行
                "。",    # 中文句子
                "！",
                "？",
                ".",
                "!",
                "?",
                " ",     # 空格
                "",      # 字符
            ],
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        print(f"将 {len(documents)} 个文档分割为 {len(chunks)} 个块")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """分割单个文本"""
        return self.splitter.split_text(text)
