"""
知识库管理 - 整合所有RAG组件
"""

import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache

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
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    MINIMAX_MODEL,
    ARK_API_KEY,
    ARK_BASE_URL,
    ARK_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_TOKENS,
    ENABLE_PREPROCESSOR,
    ENABLE_LLM_TAG,
    USE_HYBRID_RETRIEVAL,
    USE_RERANK,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RERANK_TOP_K,
    EXAMPLES,
    ENABLE_IMAGE_PROCESSING,
)
from document_loader import DocumentLoader
from document_preprocessor import DocumentPreprocessor
from test_scenario_splitter import TestScenarioSplitter
from text_splitter import TextSplitter
from vector_store import VectorStoreManager
from retriever import Retriever, AdvancedRetriever
from case_generator import TestCaseGenerator


def create_llm_provider(config: dict = None):
    """创建LLM提供商实例"""
    config = config or {}

    provider_type = config.get('llm_provider', LLM_PROVIDER)

    if provider_type == "minimax":
        from llm_provider_minimax import MiniMaxProvider
        return MiniMaxProvider(
            model=config.get('minimax_model', MINIMAX_MODEL),
            api_key=config.get('minimax_api_key', MINIMAX_API_KEY),
            base_url=config.get('minimax_base_url', MINIMAX_BASE_URL),
            max_tokens=config.get('max_tokens', MAX_TOKENS)
        )
    elif provider_type == "volcano":
        from llm_provider_volcano import VolcanoProvider
        return VolcanoProvider(
            model=config.get('ark_model', ARK_MODEL),
            api_key=config.get('ark_api_key', ARK_API_KEY),
            base_url=config.get('ark_base_url', ARK_BASE_URL),
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

    # 类级别的线程锁，用于保护共享资源
    _lock = threading.RLock()
    # 缓存已加载的检索器
    _retriever_cache: Dict[str, Retriever] = {}

    def __init__(self, config: dict = None):
        # 使用传入的配置或默认配置
        self.config = config or {}

        self.docs_dir = Path(self.config.get('docs_dir', KNOWLEDGE_BASE_DIR)).resolve()
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
        self.advanced_retriever: Optional[AdvancedRetriever] = None

    def build_knowledge_base(self, force_rebuild: bool = False):
        """构建知识库（线程安全）"""
        with self._lock:
            # 初始化增量构建器
            incremental = IncrementalBuilder()

            # 获取所有支持的文件（支持文件和文件夹）
            from document_loader import get_supported_extensions
            all_files = []

            if self.docs_dir.is_file():
                # 指定的是单个文件
                all_files = [self.docs_dir]
            else:
                # 指定的是文件夹
                for ext in get_supported_extensions():
                    all_files.extend(self.docs_dir.rglob(f'*{ext}'))

            if not all_files:
                print("未找到任何文档")
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
                # 清除缓存
                self._retriever_cache.clear()

            # 获取当前图片处理配置
            enable_image_processing = self.config.get('enable_image_processing', ENABLE_IMAGE_PROCESSING)
            print(f"图片处理配置: {'开启' if enable_image_processing else '关闭'}")

            # 获取需要处理的文件（新增或修改）
            changed_files = incremental.get_changed_files(all_files, enable_image_processing=enable_image_processing)

            # 检测已删除的文件
            deleted_files = incremental.get_deleted_files(all_files)
            if deleted_files:
                print(f"检测到 {len(deleted_files)} 个已删除的文件")
                # 先加载向量存储，然后删除这些文件的文档块
                existing_store = self.vector_manager.load_vectorstore()
                if existing_store:
                    for deleted_file in deleted_files:
                        self.vector_manager.delete_documents_by_source(deleted_file)
                        incremental.remove_deleted_file(deleted_file)

            print(f"总文件数: {len(all_files)}")
            print(f"需处理文件数: {len(changed_files)}")

            # 加载文档（只加载变化的文件）
            processed_files = []
            if changed_files:
                # 使用自定义加载函数，只加载变化的文件
                documents, processed_files = self._load_changed_files(changed_files)
            else:
                documents = []

            if not documents and not incremental.file_states:
                print("未找到任何文档，请先在docs目录下添加知识库文档")
                return False

            # 检查是否有文档需要处理（即使加载失败，也要标记文件已处理）
            has_files_to_process = len(processed_files) > 0
            
            if has_files_to_process:
                if documents:
                    # 有成功加载的文档，继续处理
                    # 预处理文档
                    if self.config.get('enable_preprocessor', ENABLE_PREPROCESSOR):
                        preprocessor = DocumentPreprocessor(
                            enable_llm=self.config.get('enable_llm_tag', ENABLE_LLM_TAG),
                            llm_provider=self.llm_provider,
                            enable_image_processing=enable_image_processing
                        )
                        documents = preprocessor.preprocess(documents)

                    # 测试场景化拆分（按章节拆分，提取测试点）
                    scenario_splitter = TestScenarioSplitter(
                        enable_llm=False,
                        llm_provider=self.llm_provider
                    )
                    documents = scenario_splitter.split_documents(documents)
                    print(f"  测试场景化拆分后: {len(documents)} 个场景切片")

                    # 分割文档
                    chunks = self.text_splitter.split_documents(documents)

                    # 检查是否已有向量存储
                    existing_store = self.vector_manager.load_vectorstore()

                    if existing_store and documents:
                        # 增量添加
                        print(f"增量添加 {len(chunks)} 个文档块...")
                        self.vector_manager.add_documents(chunks)
                        vectorstore = existing_store
                    else:
                        # 创建新的向量存储（如果有文档）
                        if documents:
                            print("创建新的向量存储...")
                            vectorstore = self.vector_manager.create_vectorstore(chunks)
                        else:
                            vectorstore = self.vector_manager.load_vectorstore()
                else:
                    # 没有成功加载的文档，但有尝试过处理文件
                    vectorstore = self.vector_manager.load_vectorstore()
                
                # 标记所有处理过的文件为已处理（包括失败的），避免下次重复尝试
                incremental.mark_processed(processed_files, enable_image_processing=enable_image_processing)
                print(f"已记录 {len(processed_files)} 个文件的处理状态")
            else:
                # 如果没有新文档需要处理，但已有知识库
                existing_store = self.vector_manager.load_vectorstore()
                if existing_store:
                    print("知识库已是最新，无需更新")
                    self.retriever = self._create_retriever(existing_store)
                    return True

            # 创建检索器并更新缓存
            self.retriever = self._create_retriever(vectorstore)
            cache_key = str(self.vector_dir)
            self._retriever_cache[cache_key] = self.retriever

            # 清除检索缓存
            if hasattr(self.retriever, 'retrieve') and hasattr(self.retriever.retrieve, 'cache_clear'):
                self.retriever.retrieve.cache_clear()
                print("检索缓存已清除")

            print("知识库构建完成！")
            return True

    def load_knowledge_base(self) -> bool:
        """加载已存在的知识库（线程安全，带缓存）"""
        with self._lock:
            # 先检查缓存
            cache_key = str(self.vector_dir)
            if cache_key in self._retriever_cache:
                self.retriever = self._retriever_cache[cache_key]
                return True

            # 如果已经加载过且retriever存在，直接返回
            if self.retriever is not None:
                return True

            vectorstore = self.vector_manager.load_vectorstore()

            if vectorstore:
                self.retriever = self._create_retriever(vectorstore)
                # 存入缓存
                self._retriever_cache[cache_key] = self.retriever
                return True

            return False

    def delete_document(self, source_pattern: str) -> int:
        """删除指定源文件的文档（线程安全）

        Args:
            source_pattern: 源文件路径或路径模式

        Returns:
            删除的文档块数量
        """
        with self._lock:
            # 从向量存储中删除
            deleted_count = self.vector_manager.delete_documents_by_source(source_pattern)

            # 从增量构建状态中移除
            incremental = IncrementalBuilder()
            incremental.remove_deleted_file(source_pattern)

            # 清除检索缓存（如果存在）
            if self.retriever and hasattr(self.retriever, 'retrieve') and hasattr(self.retriever.retrieve, 'cache_clear'):
                self.retriever.retrieve.cache_clear()

            # 清除知识库缓存，因为知识库已改变
            self._retriever_cache.clear()

            # 重新加载检索器（如果之前已加载）
            if self.retriever:
                self.load_knowledge_base()

            return deleted_count

    def clear_cache(self):
        """清除所有缓存"""
        with self._lock:
            # 清除知识库缓存
            self._retriever_cache.clear()

            # 清除检索器缓存
            if self.retriever and hasattr(self.retriever, 'retrieve') and hasattr(self.retriever.retrieve, 'cache_clear'):
                self.retriever.retrieve.cache_clear()

            print("所有缓存已清除")

    def query(self, query_text: str, return_context: bool = True, num_cases: int = 10, examples: str = None, version: str = None):
        """查询并生成测试用例

        Args:
            query_text: 查询文本
            return_context: 是否打印检索到的文档
            num_cases: 生成的测试用例数量
            examples: 用例示例，为None时从配置文件读取
            version: 可选，版本过滤，支持模糊输入（如"最新"、"第二新"、"前3个"、文件名等）
        """
        # 如果未传入examples，则从配置读取
        if examples is None:
            examples = self.config.get('examples', EXAMPLES)

        # 确保知识库已加载（如果已经加载过，load_knowledge_base会直接返回True）
        if not self.load_knowledge_base():
            raise ValueError("知识库未构建，请先运行 --build")

        # 检索相关文档（支持版本过滤）
        context_docs = self.retriever.retrieve(query_text, version=version)

        if return_context:
            self.retriever.print_retrieved_docs(context_docs)

        # 生成测试用例（支持批量）
        batch_size = min(num_cases, 36)
        result = self.test_generator.generate(
            query_text, context_docs, num_cases=num_cases, 
            batch_size=batch_size, max_retries=1, examples=examples
        )
        # 优先使用Excel格式保存
        filepath = self.test_generator.save_to_excel(result)

        return {
            'content': result,
            'filepath': str(filepath)
        }

    def check_llm_connection(self) -> bool:
        """检查LLM服务连接"""
        provider = self.config.get('llm_provider', LLM_PROVIDER)
        if provider == "minimax":
            print(f"检查 MiniMax ({MINIMAX_MODEL}) 连接...")
        elif provider == "volcano":
            print(f"检查 火山引擎 ({ARK_MODEL}) 连接...")
        else:
            print(f"检查 Ollama ({OLLAMA_MODEL}) 连接...")
        return self.llm_provider.check_connection()

    def _create_retriever(self, vectorstore) -> Retriever:
        """创建检索器（减少重复代码）"""
        return Retriever(
            vectorstore,
            top_k=self.config.get('top_k', TOP_K),
            use_hybrid=self.config.get('use_hybrid_retrieval', USE_HYBRID_RETRIEVAL),
            use_rerank=self.config.get('use_rerank', USE_RERANK),
            vector_weight=self.config.get('vector_weight', VECTOR_WEIGHT),
            bm25_weight=self.config.get('bm25_weight', BM25_WEIGHT),
            rerank_top_k=self.config.get('rerank_top_k', RERANK_TOP_K)
        )

    def _load_changed_files(self, file_paths: List[Path]) -> tuple[List, List[Path]]:
        """只加载变化的文件
        
        Returns:
            (成功加载的文档列表, 所有尝试处理过的文件列表，包括成功和失败的)
        """
        all_docs = []
        processed_files = []
        from document_loader import get_supported_extensions

        total = len(file_paths)
        print(f"开始加载 {total} 个变化的文件...")

        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{total}] 加载: {file_path.name}")
            processed_files.append(file_path)

            try:
                docs = self.document_loader.load_file(file_path)
                if docs:
                    all_docs.extend(docs)
                    print(f"成功加载: {file_path.name}")
            except Exception as e:
                print(f"加载失败 {file_path.name}: {e}")

        print(f"文件加载完成，共 {len(all_docs)} 个文档对象")
        return all_docs, processed_files

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
                llm_provider=self.llm_provider,
                enable_image_processing=self.config.get('enable_image_processing', ENABLE_IMAGE_PROCESSING)
            )
            documents = preprocessor.preprocess(documents)

        chunks = self.text_splitter.split_documents(documents)
        self.vector_manager.add_documents(chunks)

        print("文档添加完成")

    def _init_advanced_retriever(self):
        """初始化高级检索器"""
        if not self.retriever:
            self.load_knowledge_base()

        if self.advanced_retriever:
            return

        self.advanced_retriever = AdvancedRetriever(
            base_retriever=self.retriever,
            llm_provider=self.llm_provider,
            top_k=self.config.get('top_k', TOP_K),
            rerank_top_k=self.config.get('rerank_top_k', RERANK_TOP_K)
        )
        print("高级检索器初始化完成")

    def _get_examples(self, examples: str = None) -> str:
        """获取用例示例，优先使用传入值，否则从配置读取"""
        if examples is None:
            return self.config.get('examples', EXAMPLES)
        return examples

    def query_with_rewrite(self, query_text: str, return_context: bool = True, num_cases: int = 10, examples: str = None, version: str = None):
        """使用查询改写进行检索

        Args:
            query_text: 查询文本
            return_context: 是否打印检索到的文档
            num_cases: 生成的测试用例数量
            examples: 用例示例，为None时从配置文件读取
            version: 可选，版本过滤，支持模糊输入：
                - 版本号：如 "20260318_123000"
                - 关键词："最新"、"第二新"、"前3个"
                - 文档名：如 "云谷开票优化开发说明书" 或 "云谷开票"
                - 多个用逗号分隔

        Returns:
            包含结果的文件路径
        """
        examples = self._get_examples(examples)
        self._init_advanced_retriever()

        # 解析版本过滤条件（支持多个关键词，逗号分隔）
        filter_dict = None
        matched_versions = []
        boost_versions = []
        boost_doc_name = None  # 文档名优先

        if version:
            # 支持多个关键词，用逗号分隔
            keywords = [k.strip() for k in version.split(',') if k.strip()]

            # 检查是否包含纯文档名（不是版本号，不是关键词）
            for kw in keywords:
                # 使用retriever的_is_doc_name_input判断
                if self.retriever._is_doc_name_input(kw):
                    # 检查是否是版本关键词
                    version_keywords = ['最新', '最旧', '第二新', '第三新', '前', 'newest', 'oldest', 'latest']
                    is_version_keyword = any(kw == vk or kw.startswith(vk) for vk in version_keywords)

                    if not is_version_keyword:
                        # 这是文档名输入，设置为文档名优先
                        boost_doc_name = kw
                        print(f"文档名优先: 输入 '{kw}' -> 优先提取标签匹配的文档")
                        continue

                # 否则按版本号处理
                kw_versions = self.retriever.parse_version_input(kw)
                if kw_versions:
                    matched_versions.extend(kw_versions)
                    boost_versions.extend(kw_versions)
                    print(f"关键词 '{kw}' 匹配版本: {kw_versions}")
                else:
                    print(f"关键词 '{kw}' 未匹配到任何版本")

            # 去重
            if matched_versions:
                matched_versions = list(set(matched_versions))
                print(f"版本过滤: 输入 '{version}' -> 使用版本 {matched_versions}")

        # 使用多路召回，指定提升匹配版本的排名或文档名优先
        context_docs = self.advanced_retriever.multi_query_retrieve(
            query_text,
            filter=filter_dict,
            boost_versions=boost_versions,
            boost_doc_name=boost_doc_name
        )

        if return_context:
            self.retriever.print_retrieved_docs(context_docs)

        result = self.test_generator.generate(query_text, context_docs, num_cases=num_cases, examples=examples)
        filepath = self.test_generator.save_to_excel(result)

        return {
            'content': result,
            'filepath': str(filepath)
        }

    def query_with_filter(
        self,
        query_text: str,
        filter: dict,
        return_context: bool = True,
        num_cases: int = 10,
        examples: str = None
    ):
        """使用元数据过滤进行检索

        Args:
            query_text: 查询文本
            filter: 过滤条件，如 {"source": "xxx", "doc_type": "xxx"}
            return_context: 是否打印检索到的文档
            num_cases: 生成的测试用例数量
            examples: 用例示例，为None时从配置文件读取

        Returns:
            包含结果的文件路径
        """
        examples = self._get_examples(examples)
        self._init_advanced_retriever()

        context_docs = self.advanced_retriever.retrieve_with_filter(query_text, filter=filter)

        if return_context:
            self.retriever.print_retrieved_docs(context_docs)

        result = self.test_generator.generate(query_text, context_docs, num_cases=num_cases, examples=examples)
        filepath = self.test_generator.save_to_excel(result)

        return {
            'content': result,
            'filepath': str(filepath)
        }

    def query_with_context(
        self,
        query_text: str,
        history: list,
        return_context: bool = True,
        num_cases: int = 10,
        examples: str = None
    ):
        """使用多轮对话上下文进行检索

        Args:
            query_text: 当前查询文本
            history: 对话历史，格式为 [{"role": "user"/"assistant", "content": "..."}]
            return_context: 是否打印检索到的文档
            num_cases: 生成的测试用例数量
            examples: 用例示例，为None时从配置文件读取

        Returns:
            包含结果的文件路径
        """
        examples = self._get_examples(examples)
        self._init_advanced_retriever()

        context_docs = self.advanced_retriever.retrieve_with_context(query_text, history=history)

        if return_context:
            self.retriever.print_retrieved_docs(context_docs)

        result = self.test_generator.generate(query_text, context_docs, num_cases=num_cases, examples=examples)
        filepath = self.test_generator.save_to_excel(result)

        return {
            'content': result,
            'filepath': str(filepath)
        }

    def advanced_query(
        self,
        query_text: str,
        history: list = None,
        filter: dict = None,
        use_multi_query: bool = True,
        use_parent: bool = False,
        return_context: bool = True,
        num_cases: int = 10,
        examples: str = None
    ):
        """高级检索 - 综合使用多种检索策略

        Args:
            query_text: 查询文本
            history: 对话历史
            filter: 元数据过滤条件
            use_multi_query: 是否使用多路召回
            use_parent: 是否使用父文档召回
            return_context: 是否打印检索到的文档
            num_cases: 生成的测试用例数量
            examples: 用例示例，为None时从配置文件读取

        Returns:
            包含结果的文件路径
        """
        examples = self._get_examples(examples)
        self._init_advanced_retriever()

        context_docs = self.advanced_retriever.advanced_retrieve(
            query_text,
            history=history,
            filter=filter,
            use_multi_query=use_multi_query,
            use_parent=use_parent
        )

        if return_context:
            self.retriever.print_retrieved_docs(context_docs)

        result = self.test_generator.generate(query_text, context_docs, num_cases=num_cases, examples=examples)
        filepath = self.test_generator.save_to_excel(result)

        return {
            'content': result,
            'filepath': str(filepath)
        }
