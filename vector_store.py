"""
向量存储 - 使用Chroma存储和检索向量
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "test_knowledge_base"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # 初始化embedding模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectorstore: Optional[Chroma] = None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """创建向量存储"""
        print("正在创建向量存储...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name
        )

        print(f"向量存储已创建，共 {len(documents)} 个文档块")
        return self.vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """加载已存在的向量存储"""
        if not self.persist_directory.exists():
            print("向量存储目录不存在")
            return None

        # 检查是否有持久化的数据
        if not any(self.persist_directory.iterdir()):
            print("向量存储为空")
            return None

        try:
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            print("向量存储加载成功")
            return self.vectorstore
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return None

    def add_documents(self, documents: List[Document]):
        """添加文档到向量存储"""
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            print(f"已添加 {len(documents)} 个文档块")

    def delete_collection(self):
        """删除向量集合"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print(f"已删除集合: {self.collection_name}")
