"""
检索器 - 从知识库中检索相关文档
"""

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma


class Retriever:
    """检索器"""

    def __init__(self, vectorstore: Chroma, top_k: int = 3, similarity_threshold: float = None):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # 创建检索器
        search_kwargs = {"k": top_k}
        search_type = "similarity"

        if similarity_threshold:
            search_type = "similarity_score_threshold"
            search_kwargs["score_threshold"] = similarity_threshold

        self.retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def retrieve(self, query: str) -> List[Document]:
        """检索与查询相关的文档"""
        if not self.vectorstore:
            raise ValueError("向量存储未初始化")

        docs = self.retriever.invoke(query)
        return docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档（retrieve的别名）"""
        return self.retrieve(query)

    def print_retrieved_docs(self, docs: List[Document]):
        """打印检索到的文档"""
        print("\n" + "=" * 60)
        print(f"检索到 {len(docs)} 个相关文档:")
        print("=" * 60)

        for i, doc in enumerate(docs, 1):
            print(f"\n【文档 {i}】")
            print(f"来源: {doc.metadata.get('source', '未知')}")
            print(f"内容预览: {doc.page_content[:200]}...")
