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
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# 重排序
try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False

# CrossEncoder 模型缓存，避免重复加载
_cross_encoder_cache = {}
_cross_encoder_model_path = 'D:/models/bge-reranker-large'


def get_cross_encoder():
    """获取 CrossEncoder 模型实例（带缓存）"""
    if _cross_encoder_model_path not in _cross_encoder_cache:
        try:
            _cross_encoder_cache[_cross_encoder_model_path] = CrossEncoder(_cross_encoder_model_path)
            print(f"CrossEncoder 模型已加载: {_cross_encoder_model_path}")
        except Exception as e:
            print(f"CrossEncoder 模型加载失败: {e}")
            return None
    return _cross_encoder_cache[_cross_encoder_model_path]

# 父文档检索器
try:
    from langchain.retrievers import ParentDocumentRetriever
    PARENT_RETRIEVER_AVAILABLE = True
except ImportError:
    PARENT_RETRIEVER_AVAILABLE = False


# 查询改写提示词
QUERY_REWRITE_PROMPT = """你是一个查询改写专家。你的任务是将用户的问题改写为1-2个更适合语义检索的同义表达查询。

要求：
1. 严格保留原查询中的核心业务术语（如人名、模块名、功能名等），不能替换或删除
2. 只能进行同义词替换或句式调整，不能改变查询意图
3. 每个查询应该是独立且完整的
4. 只输出1-2个改写后的查询，一行一个，不要有其他内容
5. 如果原查询已经清晰表达意图，可以只输出1个

原问题：{query}

请输出1-2个改写后的查询："""

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
                corpus = [list(jieba.cut(doc.page_content)) for doc in self.all_docs]
                self.bm25 = BM25Okapi(corpus)
                print(f"BM25 初始化完成，共 {len(self.all_docs)} 篇文档")
        except Exception as e:
            print(f"BM25 初始化失败: {e}")
            self.use_hybrid = False

    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            # 使用缓存的 CrossEncoder 模型
            self.cross_encoder = get_cross_encoder()
            if self.cross_encoder:
                print("重排序模型 (bge-reranker-large) 初始化完成")
            else:
                self.use_rerank = False
        except Exception as e:
            print(f"重排序模型初始化失败: {e}")
            self.use_rerank = False

    def retrieve(self, query: str, version: str = None, file_hash: str = None) -> List[Document]:
        """检索与查询相关的文档

        Args:
            query: 查询文本
            version: 可选，按版本号过滤。支持模糊输入：
                - 空/留空：默认最新版本
                - 精确版本号：如 "20260318_123000"
                - 关键词："最新"、"第二新"、"前3个"
                - 文件名：如 "云谷开票优化开发说明书" 或 "云谷开票"
            file_hash: 可选，按文件哈希过滤（只检索指定文件的文档）

        Returns:
            相关文档列表
        """
        if not self.vectorstore:
            raise ValueError("向量存储未初始化")

        # 解析模糊版本输入
        filter_dict = {}
        doc_name_priority = None  # 文档名优先匹配的标记

        if version:
            version = version.strip()

            # 检查是否是文档名输入（可能是文件名或文档关键词）
            is_doc_name = self._is_doc_name_input(version)

            if is_doc_name:
                # 文档名输入：优先提取标签为该文档名的文档块
                doc_name_priority = version
                print(f"文档名优先匹配: 输入 '{version}' -> 优先提取标签匹配的文档")
            else:
                # 版本号输入：按原有逻辑处理
                matched_versions = self.parse_version_input(version)
                if matched_versions:
                    # 使用第一个匹配的版本（最新的）
                    filter_dict['version'] = matched_versions[0]
                    print(f"版本过滤: 输入 '{version}' -> 使用版本 '{matched_versions[0]}'")
                else:
                    print(f"版本过滤: 输入 '{version}' 未匹配到任何版本，将返回空结果")

        if file_hash:
            filter_dict['file_hash'] = file_hash

        if self.use_hybrid:
            return self._hybrid_retrieve(query, filter_dict, doc_name_priority)
        else:
            if filter_dict:
                docs = self.retriever.invoke(query, filter=filter_dict)
            else:
                docs = self.retriever.invoke(query)

            # 文档名优先：重新排序，将标签匹配的文档排在前面
            if doc_name_priority:
                docs = self._boost_by_doc_name(docs, doc_name_priority)

            return docs[:self.top_k]

    def _is_doc_name_input(self, version_input: str) -> bool:
        """判断输入是否是文档名（而非版本号）

        规则：
        1. 如果输入匹配版本号格式（纯数字/下划线/连字符，6-20位），认为是版本号
        2. 如果输入是文件名后缀（如.docx, .xlsx），认为是文件名
        3. 如果输入包含中文或普通文本，且不像是版本号，则认为是文档名
        """
        if not version_input:
            return False

        import re

        # 版本号格式：通常是 20260318_123000 或类似格式（数字为主，6-20位）
        if re.match(r'^\d[_\-\d]{5,19}$', version_input):
            return False

        # 文件后缀
        if version_input.lower() in ['.docx', '.doc', '.pdf', '.xlsx', '.xls', '.txt', '.md']:
            return True

        # 如果输入全是数字且较短（像日期但不像版本号）
        if re.match(r'^\d{4,8}$', version_input):
            return False

        # 其他情况（包含中文或普通文本），认为是文档名
        return True

    def _boost_by_doc_name(self, docs: List[Document], doc_name: str) -> List[Document]:
        """将标签中包含文档名的文档排在前面

        Args:
            docs: 文档列表
            doc_name: 文档名（可能只是部分名称）

        Returns:
            调整顺序后的文档列表
        """
        if not docs or not doc_name:
            return docs

        boosted_docs = []
        other_docs = []

        for doc in docs:
            tags = doc.metadata.get('tags', [])
            # 检查标签是否包含文档名（支持模糊匹配）
            tag_match = False
            for tag in tags:
                if doc_name in str(tag) or str(tag).startswith(doc_name) or doc_name in str(tag):
                    tag_match = True
                    break

            if tag_match:
                boosted_docs.append(doc)
                doc.metadata['doc_name_match'] = True
            else:
                other_docs.append(doc)

        # 对标签匹配的文档按重排序分数排序（降序），确保相关性高的在前
        if boosted_docs:
            boosted_docs.sort(
                key=lambda d: d.metadata.get('rerank_score', 0) if d.metadata.get('rerank_score') is not None else 0,
                reverse=True
            )

        # 匹配的文档排在前面
        result = boosted_docs + other_docs

        if boosted_docs:
            print(f"文档名优先: 标签匹配 '{doc_name}' 的文档 {len(boosted_docs)} 个排在前面")

        return result

    def _sort_by_doc_name_priority(self, docs: List[Document], doc_name: str, all_tag_matched_docs: List[Document] = None) -> List[Document]:
        """文档名优先排序

        在文档名优先模式下：
        1. 标签匹配的文档排在前面
        2. 同一文档的切片保持连续，按切片在文档中的顺序排列
        3. 不同文档之间按文档名匹配程度排序
        4. 标签不匹配的文档按混合分数排序

        Args:
            docs: 文档列表
            doc_name: 文档名
            all_tag_matched_docs: 所有标签匹配的文档（未使用，保留参数兼容性）

        Returns:
            排序后的文档列表
        """
        if not docs or not doc_name:
            return docs

        matched_docs = []
        other_docs = []

        for doc in docs:
            tags = doc.metadata.get('tags', [])
            tag_match = False
            tag_match_level = 0  # 0: 包含, 1: 前缀, 2: 完全匹配
            for tag in tags:
                tag_str = str(tag)
                if tag_str == doc_name:
                    tag_match = True
                    tag_match_level = 2
                    break
                elif tag_str.startswith(doc_name):
                    tag_match = True
                    tag_match_level = max(tag_match_level, 1)
                elif doc_name in tag_str:
                    tag_match = True
                    tag_match_level = max(tag_match_level, 0)

            if tag_match:
                matched_docs.append(doc)
                doc.metadata['doc_name_match'] = True
                doc.metadata['tag_match_level'] = tag_match_level
            else:
                other_docs.append(doc)
                doc.metadata['doc_name_match'] = False

        # 对标签匹配的文档排序：
        # 1. 先按匹配级别排序（2=完全匹配 > 1=前缀匹配 > 0=包含匹配）
        # 2. 同一匹配级别内，按源文件名排序（保持同一文档的切片在一起）
        # 3. 同一文档内，按原始顺序
        if matched_docs:
            def doc_name_sort_key(doc):
                source = doc.metadata.get('source', '')
                # 提取文档名（不含路径和版本号）
                import os
                doc_basename = os.path.basename(source)
                # 匹配级别（越高越靠前）
                level = -doc.metadata.get('tag_match_level', 0)
                return (level, doc_basename)

            matched_docs.sort(key=doc_name_sort_key)

        # 对标签不匹配的文档按混合分数排序
        if other_docs:
            other_docs.sort(
                key=lambda d: d.metadata.get('hybrid_score', 0) if d.metadata.get('hybrid_score') is not None else 0,
                reverse=True
            )

        result = matched_docs + other_docs

        print(f"文档名优先: 标签匹配 '{doc_name}' 的文档 {len(matched_docs)} 个排在前面")

        return result

    def _hybrid_retrieve(self, query: str, filter_dict: dict = None, doc_name_priority: str = None) -> List[Document]:
        """混合检索：向量 + BM25

        Args:
            query: 查询文本
            filter_dict: 可选的过滤条件字典
            doc_name_priority: 文档名优先匹配的标记
        """
        # 0. 文档名优先模式：从向量库中直接获取所有标签匹配的文档
        all_tag_matched_docs = []
        if doc_name_priority:
            all_tag_matched_docs = self._get_all_docs_by_tag(doc_name_priority)
            if all_tag_matched_docs:
                print(f"文档名优先: 从向量库直接筛选出 {len(all_tag_matched_docs)} 个标签匹配的文档")

        # 1. 向量检索（使用更大的候选集，确保标签匹配的文档能被召回）
        k = max(self.top_k * 4, 20)  # 文档名优先模式时用更大候选集
        search_kwargs = {"k": k}

        if filter_dict:
            vector_docs = self.vectorstore.similarity_search(query, filter=filter_dict, **search_kwargs)
        else:
            vector_docs = self.vectorstore.similarity_search(query, **search_kwargs)
        vector_scores = self._get_vector_scores(vector_docs, query)

        # 2. BM25 检索（BM25不支持filter，后续需要处理）
        bm25_docs, bm25_scores = self._bm25_retrieve(query, k=k)

        # 3. 合并结果
        combined = self._combine_results(
            vector_docs, vector_scores,
            bm25_docs, bm25_scores
        )

        # 3.5 文档名优先模式：补充标签匹配但未在候选集中的文档
        if doc_name_priority and all_tag_matched_docs:
            existing_contents = set(doc.page_content[:100] for doc in combined)
            for doc in all_tag_matched_docs:
                content_key = doc.page_content[:100]
                if content_key not in existing_contents:
                    combined.append(doc)
                    existing_contents.add(content_key)
            print(f"文档名优先: 补充后候选集共有 {len(combined)} 个文档")

        # 4. 如果有filter，在合并后再次过滤
        if filter_dict:
            combined = self._filter_docs(combined, filter_dict)

        # 5. 重排序
        # 注意：文档名优先模式下，跳过基于内容相关性的重排序
        # 因为用户输入文档名是想找到该文档的内容，而不是考虑查询与内容的相关性
        if self.use_rerank and len(combined) > 1 and not doc_name_priority:
            combined = self._rerank(query, combined)

        # 6. 文档名优先：按标签匹配和文件顺序排序（不使用内容相关性重排序）
        if doc_name_priority:
            combined = self._sort_by_doc_name_priority(combined, doc_name_priority, all_tag_matched_docs)

        return combined[:self.top_k]

    def _filter_docs(self, docs: List[Document], filter_dict: dict) -> List[Document]:
        """根据filter条件过滤文档

        注意：如果指定了过滤条件，会排除没有对应metadata字段的旧文档
        """
        if not filter_dict:
            return docs

        filtered = []
        for doc in docs:
            match = True
            for key, value in filter_dict.items():
                # 如果文档没有这个字段，排除该文档
                if key not in doc.metadata:
                    match = False
                    break
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered

    def _get_vector_scores(self, docs: List[Document], query: str) -> dict:
        """获取向量检索的分数"""
        scores = {}
        # 向量检索已经按相似度排序
        for i, doc in enumerate(docs):
            # 假设分数从 1 递减
            scores[doc.page_content] = 1.0 / (i + 1)
        return scores

    def _bm25_retrieve(self, query: str, k: int = None) -> Tuple[List[Document], dict]:
        """BM25 检索"""
        if not self.bm25 or not self.all_docs:
            return [], {}

        if k is None:
            k = self.top_k * 2

        # 分词查询
        query_tokens = list(jieba.cut(query))

        # 获取 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 获取 top k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        docs = []
        doc_scores = {}
        for idx in top_indices:
            if scores[idx] > 0:
                docs.append(self.all_docs[idx])
                doc_scores[self.all_docs[idx].page_content] = scores[idx]

        return docs, doc_scores

    def _get_all_docs_by_tag(self, doc_name: str) -> List[Document]:
        """从向量库中获取所有标签匹配指定文档名的文档

        Args:
            doc_name: 文档名（可能只是部分名称）

        Returns:
            标签匹配的文档列表
        """
        try:
            # 获取向量库中的所有文档
            results = self.vectorstore.get()
            if not results or 'documents' not in results:
                return []

            matched_docs = []
            for i, doc_content in enumerate(results['documents']):
                if not doc_content:
                    continue

                metadata = {}
                if 'metadatas' in results and results['metadatas']:
                    metadata = results['metadatas'][i] or {}

                tags = metadata.get('tags', [])

                # 检查标签是否包含文档名
                for tag in tags:
                    if doc_name in str(tag) or str(tag).startswith(doc_name) or doc_name in str(tag):
                        matched_docs.append(Document(
                            page_content=doc_content,
                            metadata=metadata
                        ))
                        break

            if matched_docs:
                print(f"  _get_all_docs_by_tag: 找到 {len(matched_docs)} 个标签匹配的文档")

            return matched_docs
        except Exception as e:
            print(f"  _get_all_docs_by_tag 查询失败: {e}")
            return []

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

    def get_all_versions(self) -> List[str]:
        """获取所有可用的版本列表"""
        try:
            # 获取向量存储中的所有文档
            # 使用get方法获取所有数据
            results = self.vectorstore.get()
            if not results or 'metadatas' not in results:
                return []

            versions = set()
            for metadata in results.get('metadatas', []):
                if metadata and 'version' in metadata:
                    versions.add(metadata['version'])

            # 按时间排序（降序，最新的在前）
            sorted_versions = sorted(versions, reverse=True)
            return sorted_versions
        except Exception as e:
            print(f"获取版本列表失败: {e}")
            return []

    def get_versions_by_filename(self, filename: str) -> List[str]:
        """获取指定文件的所有版本"""
        try:
            results = self.vectorstore.get()
            if not results or 'metadatas' not in results:
                return []

            versions = set()
            for metadata in results.get('metadatas', []):
                if metadata:
                    source = metadata.get('source', '')
                    if filename in source and 'version' in metadata:
                        versions.add(metadata['version'])

            sorted_versions = sorted(versions, reverse=True)
            return sorted_versions
        except Exception as e:
            print(f"获取文件版本列表失败: {e}")
            return []

    def parse_version_input(self, version_input: str) -> List[str]:
        """解析模糊版本输入，返回匹配的版本列表

        Args:
            version_input: 用户输入的版本字符串

        Returns:
            匹配的版本列表（按时间降序排列）
        """
        if not version_input or not version_input.strip():
            # 空输入，返回最新版本
            versions = self.get_all_versions()
            return versions[:1] if versions else []

        version_input = version_input.strip()

        # 获取所有版本
        all_versions = self.get_all_versions()
        if not all_versions:
            return []

        # 精确匹配版本号
        if version_input in all_versions:
            return [version_input]

        # 匹配文件名
        matching_versions = []
        for v in all_versions:
            file_versions = self.get_versions_by_filename(version_input)
            matching_versions.extend(file_versions)

        if matching_versions:
            return sorted(set(matching_versions), reverse=True)

        # 模糊匹配关键词
        version_input_lower = version_input.lower()

        # 处理特殊关键词
        if version_input_lower in ['最新', '最新版本', 'newest', 'latest']:
            return all_versions[:1]
        elif version_input_lower in ['最旧', '最早', 'oldest', 'earliest']:
            return all_versions[-1:] if all_versions else []
        elif version_input_lower in ['第二新', '倒数第二', '2nd newest']:
            return all_versions[1:2] if len(all_versions) > 1 else all_versions[:1]
        elif version_input_lower in ['第三新', '倒数第三', '3rd newest']:
            return all_versions[2:3] if len(all_versions) > 2 else all_versions[:1]

        # 处理数字前缀（如 "前3个"、"最近3个"）
        import re
        match = re.match(r'前?(\d+)个', version_input_lower)
        if match:
            count = int(match.group(1))
            return all_versions[:count]

        # 尝试模糊匹配版本号的一部分
        for v in all_versions:
            if version_input in v:
                matching_versions.append(v)

        return sorted(set(matching_versions), reverse=True) if matching_versions else []


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

        # 重排序模型（使用缓存）
        self.cross_encoder = None
        if RERANK_AVAILABLE:
            try:
                self.cross_encoder = get_cross_encoder()
            except Exception as e:
                print(f"重排序模型初始化失败: {e}")

    def rewrite_query(self, query: str) -> List[str]:
        """查询改写 - 将原始query改写为1-2个更适合检索的query

        Args:
            query: 原始查询

        Returns:
            改写后的查询列表（包含原始查询和改写查询）
        """
        if not self.llm_provider:
            print("警告: 未配置LLM提供商，无法进行查询改写")
            return [query]

        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            result = self.llm_provider.chat(prompt)

            # 解析结果，按行分割
            rewritten = [q.strip() for q in result.strip().split('\n') if q.strip()]

            # 限制最多2个改写查询
            rewritten = rewritten[:2]

            # 始终保留原始查询在第一位
            queries = [query] + rewritten

            return queries

        except Exception as e:
            print(f"查询改写失败: {e}")
            return [query, query, query]

    def multi_query_retrieve(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        use_rerank: bool = True,
        boost_versions: List[str] = None,
        boost_doc_name: str = None
    ) -> List[Document]:
        """多路召回 - 多个query分别检索，合并去重后重排序

        Args:
            query: 原始查询
            filter: 元数据过滤条件
            use_rerank: 是否使用重排序
            boost_version: 指定版本时，提升该版本文档的排名
            boost_doc_name: 指定文档名时，优先提取标签为该文档名的文档

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

        # 3. 如果指定了版本，补充检索该版本的文档
        if boost_versions:
            # 先提升已有文档的排名
            all_docs = self._boost_version_docs(all_docs, boost_versions)

            # 补充检索指定版本的文档（无论当前数量多少）
            extra_docs = self._retrieve_by_versions(query, boost_versions, self.top_k)
            # 合并并去重
            existing_contents = set(d.page_content[:100] for d in all_docs)  # 用内容前100字符作为唯一标识
            for doc in extra_docs:
                content_key = doc.page_content[:100]
                if content_key not in existing_contents:
                    all_docs.append(doc)
                    existing_contents.add(content_key)

            print(f"版本补充后总文档数: {len(all_docs)}")

        # 4. 重排序
        if use_rerank and self.cross_encoder and len(all_docs) > 1:
            all_docs = self._rerank(query, all_docs)

        # 5. 文档名优先：将标签匹配的文档排在前面（在重排序之后，确保优先级最高）
        if boost_doc_name:
            all_docs = self.base_retriever._boost_by_doc_name(all_docs, boost_doc_name)

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

    def _boost_version_docs(self, docs: List[Document], boost_versions: List[str]) -> List[Document]:
        """提升指定版本文档的排名

        Args:
            docs: 文档列表
            boost_versions: 需要提升的版本号列表

        Returns:
            调整顺序后的文档列表
        """
        if not docs or not boost_versions:
            return docs

        # 转换为集合便于快速查找
        boost_set = set(boost_versions)

        # 分离匹配版本和非匹配版本的文档
        boosted_docs = []
        other_docs = []

        for doc in docs:
            doc_version = doc.metadata.get('version', '')
            if doc_version in boost_set:
                boosted_docs.append(doc)
            else:
                other_docs.append(doc)

        # 匹配版本的文档排在前面
        result = boosted_docs + other_docs
        print(f"版本提升: 匹配版本 {boost_versions} 的文档 {len(boosted_docs)} 个排在前面")

        return result

    def _retrieve_by_versions(self, query: str, versions: List[str], limit: int = 5) -> List[Document]:
        """根据版本列表检索文档

        Args:
            query: 查询文本
            versions: 版本号列表
            limit: 最多返回数量

        Returns:
            文档列表
        """
        if not versions or not self.vectorstore:
            return []

        try:
            results = self.vectorstore.get(where={'version': {'$in': versions}})
            if not results or 'documents' not in results:
                return []

            docs = []
            for i, doc_content in enumerate(results.get('documents', [])):
                if doc_content and i < len(results.get('metadatas', [])):
                    metadata = results['metadatas'][i]
                    doc = Document(page_content=doc_content, metadata=metadata)
                    docs.append(doc)

            # 如果有重排序模型，按相关性排序
            if self.cross_encoder and docs:
                docs = self._rerank(query, docs)

            print(f"版本补充检索: 从版本 {versions} 中补充检索到 {len(docs)} 个文档")
            return docs[:limit]

        except Exception as e:
            print(f"版本补充检索失败: {e}")
            return []

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
