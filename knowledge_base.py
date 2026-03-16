"""
知识库管理 - 整合所有RAG组件
"""

import os
from pathlib import Path
from typing import List, Optional

from config import (
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
    CASES_OUTPUT_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_TOKENS,
)
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from vector_store import VectorStoreManager
from retriever import Retriever
from llm_provider import OllamaProvider
from case_generator import TestCaseGenerator


class KnowledgeBase:
    """RAG知识库管理器"""

    def __init__(self, config: dict = None):
        # 使用传入的配置或默认配置
        self.config = config or {}

        self.docs_dir = Path(self.config.get('docs_dir', KNOWLEDGE_BASE_DIR))
        self.vector_dir = Path(self.config.get('vector_dir', VECTOR_STORE_DIR))
        self.output_dir = Path(self.config.get('output_dir', CASES_OUTPUT_DIR))

        # 初始化各组件
        self.document_loader = DocumentLoader(self.docs_dir)
        self.text_splitter = TextSplitter(
            chunk_size=self.config.get('chunk_size', CHUNK_SIZE),
            chunk_overlap=self.config.get('chunk_overlap', CHUNK_OVERLAP)
        )
        self.vector_manager = VectorStoreManager(
            persist_directory=str(self.vector_dir),
            embedding_model=self.config.get('embedding_model', EMBEDDING_MODEL),
            collection_name=self.config.get('collection_name', COLLECTION_NAME)
        )
        self.llm_provider = OllamaProvider(
            model=self.config.get('ollama_model', OLLAMA_MODEL),
            base_url=self.config.get('ollama_base_url', OLLAMA_BASE_URL),
            max_tokens=self.config.get('max_tokens', MAX_TOKENS)
        )
        self.test_generator = TestCaseGenerator(
            self.llm_provider,
            self.output_dir
        )

        self.retriever: Optional[Retriever] = None

    def build_knowledge_base(self, force_rebuild: bool = False):
        """构建知识库"""
        # 加载文档
        documents = self.document_loader.load_directory()

        if not documents:
            print("未找到任何文档，请先在docs目录下添加知识库文档")
            return False

        # 检查是否需要重建
        existing_store = self.vector_manager.load_vectorstore()
        if existing_store and not force_rebuild:
            print("知识库已存在，使用 --build --rebuild 强制重建")
            self.retriever = Retriever(
                existing_store,
                top_k=self.config.get('top_k', TOP_K)
            )
            return True

        # 分割文档
        chunks = self.text_splitter.split_documents(documents)

        # 创建向量存储
        vectorstore = self.vector_manager.create_vectorstore(chunks)

        # 创建检索器
        self.retriever = Retriever(
            vectorstore,
            top_k=self.config.get('top_k', TOP_K)
        )

        print("知识库构建完成！")
        return True

    def load_knowledge_base(self) -> bool:
        """加载已存在的知识库"""
        vectorstore = self.vector_manager.load_vectorstore()

        if vectorstore:
            self.retriever = Retriever(
                vectorstore,
                top_k=self.config.get('top_k', TOP_K)
            )
            return True

        return False

    def query(self, query_text: str, return_context: bool = True):
        """查询并生成测试用例"""
        # 确保知识库已加载
        if not self.load_knowledge_base():
            raise ValueError("知识库未构建，请先运行 --build")

        # 检索相关文档
        context_docs = self.retriever.retrieve(query_text)

        if return_context:
            self.retriever.print_retrieved_docs(context_docs)

        # 生成测试用例
        result = self.test_generator.generate_and_save(query_text, context_docs)

        return result

    def check_ollama_connection(self) -> bool:
        """检查Ollama连接"""
        return self.llm_provider.check_connection()

    def add_documents(self, docs_dir: Path = None):
        """添加新文档到知识库"""
        if docs_dir:
            self.document_loader = DocumentLoader(docs_dir)

        documents = self.document_loader.load_directory()

        if not documents:
            print("未找到新文档")
            return

        chunks = self.text_splitter.split_documents(documents)
        self.vector_manager.add_documents(chunks)

        print("文档添加完成")
