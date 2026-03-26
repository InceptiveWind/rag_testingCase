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
    USE_HYBRID_RETRIEVAL,
    USE_RERANK,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RERANK_TOP_K,
    EXAMPLES,
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
        self.advanced_retriever: Optional[AdvancedRetriever] = None

    def build_knowledge_base(self, force_rebuild: bool = False):
        """构建知识库"""
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

        # 获取当前图片处理配置
        enable_image_processing = self.config.get('enable_image_processing', ENABLE_IMAGE_PROCESSING)
        print(f"图片处理配置: {'开启' if enable_image_processing else '关闭'}")

        # 获取需要处理的文件（新增或修改）
        changed_files = incremental.get_changed_files(all_files, enable_image_processing=enable_image_processing)

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
                self.retriever = self._create_retriever(existing_store)
                return True

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

        if existing_store and changed_files:
            # 增量添加
            print(f"增量添加 {len(chunks)} 个文档块...")
            self.vector_manager.add_documents(chunks)
            vectorstore = existing_store
        else:
            # 创建新的向量存储
            print("创建新的向量存储...")
            vectorstore = self.vector_manager.create_vectorstore(chunks)

        # 标记文件已处理（记录图片处理状态）
        incremental.mark_processed(changed_files, enable_image_processing=enable_image_processing)

        # 创建检索器
        self.retriever = self._create_retriever(vectorstore)

        print("知识库构建完成！")
        return True

    def load_knowledge_base(self) -> bool:
        """加载已存在的知识库"""
        # 如果已经加载过且retriever存在，直接返回
        if self.retriever is not None:
            return True

        vectorstore = self.vector_manager.load_vectorstore()

        if vectorstore:
            self.retriever = self._create_retriever(vectorstore)
            return True

        return False

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
        result = self.test_generator.generate(query_text, context_docs, num_cases=num_cases, examples=examples)
        # 优先使用Excel格式保存
        filepath = self.test_generator.save_to_excel(result)

        return {
            'content': result,
            'filepath': str(filepath)
        }

    def check_llm_connection(self) -> bool:
        """检查LLM服务连接"""
        provider = self.config.get('llm_provider', LLM_PROVIDER)
        if provider == "volcano":
            print(f"检查火山引擎 ({VOLCANO_MODEL}) 连接...")
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
