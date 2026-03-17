"""
检索器 - 从知识库中检索相关文档
支持 BM25 + 向量混合检索 + 重排序 + 高级检索功能
"""

import os
from typing import List, Tuple, Dict, Optional, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma

# BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# 重排序
try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False

# 父文档检索器
try:
    from langchain.retrievers import ParentDocumentRetriever
    PARENT_RETRIEVER_AVAILABLE = True
except ImportError:
    PARENT_RETRIEVER_AVAILABLE = False


# 查询改写提示词
QUERY_REWRITE_PROMPT = """你是一个查询改写专家。你的任务是将用户的问题改写为3个不同的、更适合语义检索的查询。

要求：
1. 改写的查询应该涵盖原问题的不同角度和意图
2. 使用不同的关键词和表达方式
3. 每个查询应该是独立且完整的
4. 只输出3个查询，一行一个，不要有其他内容

原问题：{query}

请输出3个改写后的查询："""

# 多轮对话上下文处理提示词
CONTEXT_QUERY_PROMPT = """你是一个查询处理专家。根据对话历史，将当前问题改写为一个独立的、完整的查询。

对话历史：
{history}

当前问题：{query}

要求：
1. 结合上下文信息补充当前问题的关键信息
2. 生成的查询应该包含所有必要的上下文
3. 只输出改写后的查询，不要有其他内容

改写后的查询："""


class Retriever:
    """检索器 - 支持混合检索和重排序"""

    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = 3,
        similarity_threshold: float = None,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        rerank_top_k: int = 10
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # 混合检索配置
        self.use_hybrid = use_hybrid and BM25_AVAILABLE
        self.use_rerank = use_rerank and RERANK_AVAILABLE
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # 重排序候选数量
        self.rerank_top_k = rerank_top_k

        # 获取所有文档用于 BM25
        self.all_docs: List[Document] = []
        self.bm25 = None

        if self.use_hybrid:
            self._init_bm25()

        # 初始化重排序模型
        self.cross_encoder = None
        if self.use_rerank:
            self._init_reranker()

        # 创建向量检索器
        search_kwargs = {"k": top_k * 2}  # 多取一些用于混合
        search_type = "similarity"

        if similarity_threshold:
            search_type = "similarity_score_threshold"
            search_kwargs["score_threshold"] = similarity_threshold

        self.retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def _init_bm25(self):
        """初始化 BM25"""
        try:
            # 从向量存储获取所有文档
            collection = self.vectorstore.get()
            if not collection or not collection.get('documents'):
                print("警告: 向量存储为空，无法初始化 BM25")
                return

            # 构建文档列表
            self.all_docs = []
            for i, doc_content in enumerate(collection['documents']):
                if doc_content:
                    metadata = {}
                    if 'metadatas' in collection and collection['metadatas']:
                        metadata = collection['metadatas'][i] or {}

                    self.all_docs.append(Document(
                        page_content=doc_content,
                        metadata=metadata
                    ))

            if self.all_docs:
                # 分词处理
                corpus = [doc.page_content.split() for doc in self.all_docs]
                self.bm25 = BM25Okapi(corpus)
                print(f"BM25 初始化完成，共 {len(self.all_docs)} 篇文档")
        except Exception as e:
            print(f"BM25 初始化失败: {e}")
            self.use_hybrid = False

    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            # 使用 BGE reranker large 模型
            self.cross_encoder = CrossEncoder('D:/models/bge-reranker-large')
            print("重排序模型 (bge-reranker-large) 初始化完成")
        except Exception as e:
            print(f"重排序模型初始化失败: {e}")
            self.use_rerank = False

    def retrieve(self, query: str) -> List[Document]:
        """检索与查询相关的文档"""
        if not self.vectorstore:
            raise ValueError("向量存储未初始化")

        if self.use_hybrid:
            return self._hybrid_retrieve(query)
        else:
            docs = self.retriever.invoke(query)
            return docs[:self.top_k]

    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """混合检索：向量 + BM25"""
        # 1. 向量检索
        vector_docs = self.retriever.invoke(query)
        vector_scores = self._get_vector_scores(vector_docs, query)

        # 2. BM25 检索
        bm25_docs, bm25_scores = self._bm25_retrieve(query)

        # 3. 合并结果
        combined = self._combine_results(
            vector_docs, vector_scores,
            bm25_docs, bm25_scores
        )

        # 4. 重排序
        if self.use_rerank and len(combined) > 1:
            combined = self._rerank(query, combined)

        return combined[:self.top_k]

    def _get_vector_scores(self, docs: List[Document], query: str) -> dict:
        """获取向量检索的分数"""
        scores = {}
        # 向量检索已经按相似度排序
        for i, doc in enumerate(docs):
            # 假设分数从 1 递减
            scores[doc.page_content] = 1.0 / (i + 1)
        return scores

    def _bm25_retrieve(self, query: str) -> Tuple[List[Document], dict]:
        """BM25 检索"""
        if not self.bm25 or not self.all_docs:
            return [], {}

        # 分词查询
        query_tokens = query.split()

        # 获取 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 获取 top k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k * 2]

        docs = []
        doc_scores = {}
        for idx in top_indices:
            if scores[idx] > 0:
                docs.append(self.all_docs[idx])
                doc_scores[self.all_docs[idx].page_content] = scores[idx]

        return docs, doc_scores

    def _combine_results(
        self,
        vector_docs: List[Document], vector_scores: dict,
        bm25_docs: List[Document], bm25_scores: dict
    ) -> List[Document]:
        """合并向量和 BM25 的结果"""
        # 归一化分数
        max_vector_score = max(vector_scores.values()) if vector_scores else 1.0
        max_bm25_score = max(bm25_scores.values()) if bm25_scores else 1.0

        # 计算综合分数
        doc_scores = {}
        all_docs_dict = {}

        # 向量分数
        for doc in vector_docs:
            content = doc.page_content
            norm_score = vector_scores.get(content, 0) / max_vector_score if max_vector_score > 0 else 0
            doc_scores[content] = doc_scores.get(content, 0) + norm_score * self.vector_weight
            all_docs_dict[content] = doc

        # BM25 分数
        for doc in bm25_docs:
            content = doc.page_content
            norm_score = bm25_scores.get(content, 0) / max_bm25_score if max_bm25_score > 0 else 0
            doc_scores[content] = doc_scores.get(content, 0) + norm_score * self.bm25_weight
            all_docs_dict[content] = doc

        # 按分数排序
        sorted_contents = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回排序后的文档
        result = []
        seen = set()
        for content, score in sorted_contents:
            if content not in seen:
                doc = all_docs_dict[content]
                # 添加分数到 metadata
                doc.metadata['hybrid_score'] = score
                result.append(doc)
                seen.add(content)

        return result

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """使用交叉编码器重排序"""
        if not self.cross_encoder or not docs:
            return docs

        try:
            # 准备句子对
            pairs = [[query, doc.page_content] for doc in docs]

            # 获取重排序分数
            scores = self.cross_encoder.predict(pairs)

            # 按分数排序
            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # 返回重排序后的文档
            result = []
            for doc, score in doc_scores:
                doc.metadata['rerank_score'] = float(score)
                result.append(doc)

            return result
        except Exception as e:
            print(f"重排序失败: {e}")
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
            score_info = ""
            if 'hybrid_score' in doc.metadata:
                score_info = f" [混合分数: {doc.metadata['hybrid_score']:.3f}]"
            if 'rerank_score' in doc.metadata:
                score_info = f" [重排分数: {doc.metadata['rerank_score']:.3f}]"

            print(f"\n【文档 {i}】{score_info}")
            print(f"来源: {doc.metadata.get('source', '未知')}")
            print(f"内容预览: {doc.page_content[:200]}...")


class AdvancedRetriever:
    """高级检索器 - 支持查询改写、多路召回、父文档召回、元数据过滤、多轮对话"""

    def __init__(
        self,
        base_retriever: Retriever,
        llm_provider: Any = None,
        parent_vectorstore: Chroma = None,
        child_vectorstore: Chroma = None,
        parent_text_splitter=None,
        child_text_splitter=None,
        top_k: int = 5,
        rerank_top_k: int = 10
    ):
        """初始化高级检索器

        Args:
            base_retriever: 基础检索器
            llm_provider: LLM提供商（用于查询改写和多轮对话）
            parent_vectorstore: 父文档向量存储（用于父文档召回）
            child_vectorstore: 子文档向量存储（用于父文档召回）
            parent_text_splitter: 父文档分块器（用于父文档召回）
            child_text_splitter: 子文档分块器（用于父文档召回）
            top_k: 检索top-k
            rerank_top_k: 重排序候选数量
        """
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider
        self.vectorstore = base_retriever.vectorstore
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

        # 父文档召回相关
        self.parent_vectorstore = parent_vectorstore
        self.child_vectorstore = child_vectorstore
        self.parent_text_splitter = parent_text_splitter
        self.child_text_splitter = child_text_splitter

        # 重排序模型
        self.cross_encoder = None
        if RERANK_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder('D:/models/bge-reranker-large')
            except Exception as e:
                print(f"重排序模型初始化失败: {e}")

    def rewrite_query(self, query: str) -> List[str]:
        """查询改写 - 将原始query改写为3个更适合检索的query

        Args:
            query: 原始查询

        Returns:
            改写后的3个查询列表
        """
        if not self.llm_provider:
            print("警告: 未配置LLM提供商，无法进行查询改写")
            return [query]

        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            result = self.llm_provider.chat(prompt)

            # 解析结果，按行分割
            queries = [q.strip() for q in result.strip().split('\n') if q.strip()]

            # 确保返回3个查询
            if len(queries) >= 3:
                return queries[:3]
            else:
                # 如果返回不足3个，补充原始查询
                while len(queries) < 3:
                    queries.append(query)
                return queries[:3]

        except Exception as e:
            print(f"查询改写失败: {e}")
            return [query, query, query]

    def multi_query_retrieve(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        use_rerank: bool = True
    ) -> List[Document]:
        """多路召回 - 3个query分别检索，合并去重后重排序

        Args:
            query: 原始查询
            filter: 元数据过滤条件
            use_rerank: 是否使用重排序

        Returns:
            检索到的文档列表
        """
        # 1. 查询改写
        rewritten_queries = self.rewrite_query(query)
        print(f"原始查询: {query}")
        print(f"改写查询: {rewritten_queries}")

        # 2. 分别检索
        all_docs = []
        doc_map = {}  # 用于去重

        for q in rewritten_queries:
            try:
                docs = self.base_retriever.retrieve(q)
                for doc in docs:
                    # 使用 doc_id 或 source+内容hash 作为唯一标识
                    doc_id = doc.metadata.get('doc_id') or hash(doc.page_content)
                    if doc_id not in doc_map:
                        doc_map[doc_id] = doc
                        all_docs.append(doc)
            except Exception as e:
                print(f"检索查询 '{q}' 失败: {e}")

        # 3. 重排序
        if use_rerank and self.cross_encoder and len(all_docs) > 1:
            all_docs = self._rerank(query, all_docs)

        return all_docs[:self.top_k]

    def retrieve_with_filter(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        use_rerank: bool = True
    ) -> List[Document]:
        """元数据过滤检索

        Args:
            query: 查询内容
            filter: 过滤条件，支持的字段：source, doc_type, created_time, permissions等
            use_rerank: 是否使用重排序

        Returns:
            过滤后的文档列表
        """
        if not filter:
            return self.base_retriever.retrieve(query)

        try:
            # 使用向量存储的过滤功能
            docs = self.vectorstore.similarity_search(
                query,
                k=self.top_k * 2,
                filter=filter
            )

            # 重排序
            if use_rerank and self.cross_encoder and len(docs) > 1:
                docs = self._rerank(query, docs)

            return docs[:self.top_k]

        except Exception as e:
            print(f"过滤检索失败: {e}")
            return self.base_retriever.retrieve(query)

    def retrieve_with_context(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
        use_rewrite: bool = True,
        use_rerank: bool = True
    ) -> List[Document]:
        """多轮对话上下文处理检索

        Args:
            query: 当前问题
            history: 对话历史，格式为 [{"role": "user"/"assistant", "content": "..."}]
            use_rewrite: 是否使用查询改写
            use_rerank: 是否使用重排序

        Returns:
            检索到的文档列表
        """
        if not history:
            # 无历史记录，直接检索
            if use_rewrite and self.llm_provider:
                return self.multi_query_retrieve(query, use_rerank=use_rerank)
            else:
                return self.base_retriever.retrieve(query)

        # 有历史记录，结合上下文改写查询
        if self.llm_provider:
            try:
                # 格式化历史记录
                history_text = ""
                for msg in history:
                    role = "用户" if msg["role"] == "user" else "助手"
                    history_text += f"{role}: {msg['content']}\n"

                # 生成上下文查询
                prompt = CONTEXT_QUERY_PROMPT.format(
                    history=history_text,
                    query=query
                )
                contextual_query = self.llm_provider.chat(prompt).strip()
                print(f"上下文查询: {contextual_query}")

                # 使用改写后的查询检索
                if use_rewrite:
                    return self.multi_query_retrieve(contextual_query, use_rerank=use_rerank)
                else:
                    docs = self.base_retriever.retrieve(contextual_query)
                    if use_rerank and self.cross_encoder and len(docs) > 1:
                        docs = self._rerank(contextual_query, docs)
                    return docs[:self.top_k]

            except Exception as e:
                print(f"上下文处理失败: {e}")

        # 如果LLM不可用，直接使用原始查询
        return self.base_retriever.retrieve(query)

    def retrieve_with_parent(
        self,
        query: str,
        child_filter: Optional[Dict[str, Any]] = None,
        use_rerank: bool = True
    ) -> List[Document]:
        """父文档召回 - 检索小语义块返回完整父块

        Args:
            query: 查询内容
            child_filter: 子文档过滤条件
            use_rerank: 是否使用重排序

        Returns:
            父文档列表
        """
        if not PARENT_RETRIEVER_AVAILABLE:
            print("警告: ParentDocumentRetriever不可用，使用普通检索")
            return self.base_retriever.retrieve(query)

        if not self.parent_vectorstore or not self.child_vectorstore:
            print("警告: 未配置父子向量存储，使用普通检索")
            return self.base_retriever.retrieve(query)

        try:
            # 创建父文档检索器
            parent_retriever = ParentDocumentRetriever(
                vectorstore=self.parent_vectorstore,
                docstore=self.child_vectorstore,
                parent_splitter=self.parent_text_splitter,
                child_splitter=self.child_text_splitter,
                search_kwargs={"k": self.top_k}
            )

            docs = parent_retriever.invoke(query)

            # 重排序
            if use_rerank and self.cross_encoder and len(docs) > 1:
                docs = self._rerank(query, docs)

            return docs[:self.top_k]

        except Exception as e:
            print(f"父文档召回失败: {e}")
            return self.base_retriever.retrieve(query)

    def advanced_retrieve(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        use_multi_query: bool = True,
        use_parent: bool = False,
        use_rerank: bool = True
    ) -> List[Document]:
        """高级检索 - 综合使用多种检索策略

        Args:
            query: 查询内容
            history: 对话历史
            filter: 元数据过滤条件
            use_multi_query: 是否使用多路召回
            use_parent: 是否使用父文档召回
            use_rerank: 是否使用重排序

        Returns:
            检索到的文档列表
        """
        # 1. 优先处理多轮对话上下文
        if history and self.llm_provider:
            contextual_query = self._get_contextual_query(query, history)
        else:
            contextual_query = query

        # 2. 选择检索策略
        if use_parent:
            # 父文档召回
            docs = self.retrieve_with_parent(contextual_query, use_rerank=use_rerank)
        elif use_multi_query and self.llm_provider:
            # 多路召回
            docs = self.multi_query_retrieve(contextual_query, filter=filter, use_rerank=use_rerank)
        elif filter:
            # 过滤检索
            docs = self.retrieve_with_filter(contextual_query, filter=filter, use_rerank=use_rerank)
        else:
            # 普通检索
            docs = self.base_retriever.retrieve(contextual_query)

        return docs[:self.top_k]

    def _get_contextual_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """获取结合上下文的查询"""
        try:
            history_text = ""
            for msg in history:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"

            prompt = CONTEXT_QUERY_PROMPT.format(
                history=history_text,
                query=query
            )
            return self.llm_provider.chat(prompt).strip()

        except Exception as e:
            print(f"上下文查询生成失败: {e}")
            return query

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """使用交叉编码器重排序"""
        if not self.cross_encoder or not docs:
            return docs

        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.cross_encoder.predict(pairs)

            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            result = []
            for doc, score in doc_scores:
                doc.metadata['rerank_score'] = float(score)
                result.append(doc)

            return result
        except Exception as e:
            print(f"重排序失败: {e}")
            return docs
