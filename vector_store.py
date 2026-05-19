"""
向量存储 - 使用Chroma存储和检索向量
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


MAX_BATCH_SIZE = 5000  # ChromaDB批处理上限，留出余量


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

    def _batch_documents(self, documents: List[Document]) -> List[List[Document]]:
        """将文档列表分批"""
        return [documents[i:i + MAX_BATCH_SIZE] for i in range(0, len(documents), MAX_BATCH_SIZE)]

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """创建向量存储"""
        print("正在创建向量存储...")

        batches = self._batch_documents(documents)
        print(f"文档分批: {len(batches)} 批")

        for i, batch in enumerate(batches):
            print(f"正在处理第 {i+1}/{len(batches)} 批 ({len(batch)} 个文档)...")
            if i == 0:
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embedding_model,
                    persist_directory=str(self.persist_directory),
                    collection_name=self.collection_name
                )
            else:
                self.vectorstore.add_documents(batch)

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
        """添加文档到向量存储（分批处理）"""
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore:
            batches = self._batch_documents(documents)
            total = 0
            for i, batch in enumerate(batches):
                self.vectorstore.add_documents(batch)
                total += len(batch)
                print(f"已添加第 {i+1}/{len(batches)} 批，当前累计 {total} 个文档块")

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """根据源文件路径删除文档

        Args:
            source_pattern: 源文件路径或路径模式

        Returns:
            删除的文档数量
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        if not self.vectorstore:
            return 0

        try:
            # 获取所有文档，查找匹配 source 的
            results = self.vectorstore.get()
            if not results or not results.get('ids'):
                return 0

            ids_to_delete = []
            for i, metadata in enumerate(results.get('metadatas', [])):
                if metadata:
                    source = metadata.get('source', '')
                    if source_pattern in source:
                        ids_to_delete.append(results['ids'][i])

            if ids_to_delete:
                self.vectorstore.delete(ids=ids_to_delete)
                print(f"已删除 {len(ids_to_delete)} 个文档块（来源: {source_pattern}）")
                return len(ids_to_delete)

            return 0
        except Exception as e:
            print(f"删除文档失败: {e}")
            return 0

    def delete_all_documents(self):
        """删除所有文档"""
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore:
            try:
                # 获取所有文档ID并删除
                results = self.vectorstore.get()
                if results and results.get('ids'):
                    self.vectorstore.delete(ids=results['ids'])
                    print(f"已删除所有文档（共 {len(results['ids'])} 个）")
            except Exception as e:
                print(f"删除所有文档失败: {e}")

