"""
知识库管理 - 整合所有RAG组件
"""

import os
from pathlib import Path
from typing import List, Optional

# 设置环境变量解决并行问题
os.environ['TOKENIZERS_PARALLELISM'] = '0'

from incremental_builder import IncrementalBuilder

from config import (
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
    CASES_OUTPUT_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_TOKENS,
    ENABLE_PREPROCESSOR,
    ENABLE_LLM_TAG,
)
from document_loader import DocumentLoader
from document_preprocessor import DocumentPreprocessor
from text_splitter import TextSplitter
from vector_store import VectorStoreManager
from retriever import Retriever
from case_generator import TestCaseGenerator


def create_llm_provider(config: dict = None):
    """创建LLM提供商实例"""
    config = config or {}

    provider_type = config.get('llm_provider', LLM_PROVIDER)

    if provider_type == "volcano":
        from llm_provider_volcano import VolcanoProvider
        return VolcanoProvider(
            model=config.get('volcano_model', VOLCANO_MODEL),
            api_key=config.get('volcano_api_key', VOLCANO_API_KEY),
            base_url=config.get('volcano_base_url', VOLCANO_BASE_URL),
            max_tokens=config.get('max_tokens', MAX_TOKENS)
        )
    else:
        from llm_provider import OllamaProvider
        return OllamaProvider(
            model=config.get('ollama_model', OLLAMA_MODEL),
            base_url=config.get('ollama_base_url', OLLAMA_BASE_URL),
            max_tokens=config.get('max_tokens', MAX_TOKENS)
        )


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
        # 创建LLM提供商
        self.llm_provider = create_llm_provider(self.config)
        self.test_generator = TestCaseGenerator(
            self.llm_provider,
            self.output_dir
        )

        self.retriever: Optional[Retriever] = None

    def build_knowledge_base(self, force_rebuild: bool = False):
        """构建知识库"""
        # 初始化增量构建器
        incremental = IncrementalBuilder()

        # 获取所有支持的文件
        from document_loader import get_supported_extensions
        all_files = []
        for ext in get_supported_extensions():
            all_files.extend(self.docs_dir.rglob(f'*{ext}'))

        if not all_files:
            print("未找到任何文档，请先在docs目录下添加知识库文档")
            return False

        # 强制重建：清除所有状态
        if force_rebuild:
            print("强制重建，清除历史状态...")
            incremental.clear()
            # 删除向量存储
            if self.vector_dir.exists():
                import shutil
                shutil.rmtree(self.vector_dir)
                self.vector_dir.mkdir(parents=True, exist_ok=True)

        # 获取需要处理的文件（新增或修改）
        changed_files = incremental.get_changed_files(all_files)

        print(f"总文件数: {len(all_files)}")
        print(f"需处理文件数: {len(changed_files)}")

        # 加载文档（只加载变化的文件）
        if changed_files:
            # 使用自定义加载函数，只加载变化的文件
            documents = self._load_changed_files(changed_files)
        else:
            documents = []

        if not documents and not incremental.file_states:
            print("未找到任何文档，请先在docs目录下添加知识库文档")
            return False

        # 如果没有新文档需要处理，但已有知识库
        if not documents:
            existing_store = self.vector_manager.load_vectorstore()
            if existing_store:
                print("知识库已是最新，无需更新")
                self.retriever = Retriever(
                    existing_store,
                    top_k=self.config.get('top_k', TOP_K)
                )
                return True

        # 预处理文档
        if self.config.get('enable_preprocessor', ENABLE_PREPROCESSOR):
            preprocessor = DocumentPreprocessor(
                enable_llm=self.config.get('enable_llm_tag', ENABLE_LLM_TAG),
                llm_provider=self.llm_provider
            )
            documents = preprocessor.preprocess(documents)

        # 分割文档
        chunks = self.text_splitter.split_documents(documents)

        # 检查是否已有向量存储
        existing_store = self.vector_manager.load_vectorstore()

        if existing_store and changed_files:
            # 增量添加
            print(f"增量添加 {len(chunks)} 个文档块...")
            self.vector_manager.add_documents(chunks)
            vectorstore = existing_store
        else:
            # 创建新的向量存储
            print("创建新的向量存储...")
            vectorstore = self.vector_manager.create_vectorstore(chunks)

        # 标记文件已处理
        incremental.mark_processed(changed_files)

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

    def check_llm_connection(self) -> bool:
        """检查LLM服务连接"""
        provider = self.config.get('llm_provider', LLM_PROVIDER)
        if provider == "volcano":
            print(f"检查火山引擎 ({VOLCANO_MODEL}) 连接...")
        else:
            print(f"检查 Ollama ({OLLAMA_MODEL}) 连接...")
        return self.llm_provider.check_connection()

    def _load_changed_files(self, file_paths: List[Path]) -> List:
        """只加载变化的文件"""
        all_docs = []
        from document_loader import get_supported_extensions

        total = len(file_paths)
        print(f"开始加载 {total} 个变化的文件...")

        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{total}] 加载: {file_path.name}")

            try:
                docs = self.document_loader.load_file(file_path)
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                print(f"加载失败 {file_path.name}: {e}")

        print(f"文件加载完成，共 {len(all_docs)} 个文档对象")
        return all_docs

    def add_documents(self, docs_dir: Path = None):
        """添加新文档到知识库"""
        if docs_dir:
            self.document_loader = DocumentLoader(docs_dir)

        documents = self.document_loader.load_directory()

        if not documents:
            print("未找到新文档")
            return

        # 预处理文档
        if self.config.get('enable_preprocessor', ENABLE_PREPROCESSOR):
            preprocessor = DocumentPreprocessor(
                enable_llm=self.config.get('enable_llm_tag', ENABLE_LLM_TAG),
                llm_provider=self.llm_provider
            )
            documents = preprocessor.preprocess(documents)

        chunks = self.text_splitter.split_documents(documents)
        self.vector_manager.add_documents(chunks)

        print("文档添加完成")
