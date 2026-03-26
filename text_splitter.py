"""
文本分块器 - 支持多级语义分块
"""

import re
from typing import List, Set, Tuple
from langchain_core.documents import Document


class SemanticTextSplitter:
    """语义分块器 - 按句子/段落分块，保持语义完整性"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300, min_chunk_length: int = 50):
        """
        Args:
            chunk_size: 块的最大字符数
            chunk_overlap: 相邻块的重叠字符数
            min_chunk_length: 最小块长度（过短的块会合并）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        # 句子边界模式（中文和英文）
        self.sentence_endings = re.compile(r'[。！？.!?]+')
        # 测试用例关键结构模式（应保持完整）
        self.test_case_patterns = [
            r'用例名称[：:][^\n]+',
            r'前置条件[：:][^\n]+',
            r'优先级[：:][^\n]+',
            r'步骤ID[：:][^\n]+',
            r'测试步骤[：:][^\n]+',
            r'预期结果[：:][^\n]+',
            r'\d+[、.、]((?:(?!预期结果).)+)',  # 匹配步骤但不包含预期结果
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        if not documents:
            return []

        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        print(f"将 {len(documents)} 个文档分割为 {len(all_chunks)} 个语义块")
        return all_chunks

    def split_document(self, doc: Document) -> List[Document]:
        """分割单个文档为语义块"""
        content = doc.page_content
        metadata = doc.metadata.copy()

        # 检测是否为测试用例文档
        is_test_case = self._is_test_case_content(content)

        # 第一级：按段落分割
        paragraphs = self._split_by_paragraphs(content)

        chunks = []
        current_chunk = ""
        current_sources = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果是测试用例关键结构，检查是否需要保持完整
            if is_test_case and self._is_test_case_key_structure(para):
                # 先保存当前的chunk
                if current_chunk:
                    chunks.extend(self._create_chunks(current_chunk, metadata))
                    current_chunk = ""

                # 检查整个关键结构是否超出限制
                if len(para) > self.chunk_size * 1.5:
                    # 超出太多，按句子分割但保持关键结构
                    chunks.extend(self._split_with_structure_preserved(para, metadata))
                else:
                    # 保持完整
                    chunks.append(self._create_chunk_doc(para, metadata))
            elif len(current_chunk) + len(para) + 2 <= self.chunk_size:
                # 可以加入当前chunk
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                # 超出限制，保存当前chunk并开始新的
                if current_chunk:
                    chunks.extend(self._create_chunks(current_chunk, metadata))
                current_chunk = para

        # 处理最后一个chunk
        if current_chunk:
            chunks.extend(self._create_chunks(current_chunk, metadata))

        # 标记chunk类型
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_type'] = self._detect_chunk_type(chunk.page_content)

        return chunks

    def _is_test_case_content(self, content: str) -> bool:
        """检测内容是否为测试用例相关"""
        test_keywords = ['用例名称', '测试步骤', '预期结果', '前置条件', '步骤ID', '优先级']
        return any(kw in content for kw in test_keywords)

    def _is_test_case_key_structure(self, text: str) -> bool:
        """检测是否为测试用例关键结构（应保持完整）"""
        for pattern in self.test_case_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _split_by_paragraphs(self, content: str) -> List[str]:
        """按段落分割"""
        # 按双换行或单换行分割
        paragraphs = re.split(r'\n\s*\n|\n', content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_with_structure_preserved(self, text: str, metadata: dict) -> List[Document]:
        """保持结构地分割（用于测试用例关键结构）"""
        chunks = []

        # 尝试按"步骤"分割
        steps = re.split(r'(步骤ID[：:]\d*[^\n]*(?:\n(?:(?!(?:步骤ID|预期结果|前置条件))[^\n])*)*)', text)

        current = ""
        for part in steps:
            if not part.strip():
                continue

            if len(current) + len(part) + 2 <= self.chunk_size:
                if current:
                    current += "\n"
                current += part
            else:
                if current:
                    chunks.append(self._create_chunk_doc(current, metadata))
                # 如果单个步骤就超出限制，按句子继续分割
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_by_sentences_only(part, metadata)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(self._create_chunk_doc(current, metadata))

        return chunks

    def _split_by_sentences_only(self, text: str, metadata: dict) -> List[Document]:
        """仅按句子分割（不合并）"""
        sentences = self._split_into_sentences(text)
        result = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    result.append(self._create_chunk_doc(current, metadata))
                current = sent

        if current:
            result.append(self._create_chunk_doc(current, metadata))

        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        sentences = []
        current = ""
        i = 0
        while i < len(text):
            char = text[i]
            current += char

            # 检查是否为句子结尾
            if char in '。！？.!?':
                sentences.append(current)
                current = ""
            elif char in '\n':
                # 换行也可以作为分割点
                sentences.append(current)
                current = ""

            i += 1

        if current.strip():
            sentences.append(current)

        return sentences

    def _create_chunks(self, text: str, metadata: dict) -> List[Document]:
        """创建chunks，处理重叠和过短问题"""
        if not text.strip():
            return []

        # 检查是否需要分割
        if len(text) <= self.chunk_size:
            # 检查是否过短，过短则不单独创建
            if len(text) < self.min_chunk_length:
                return []
            return [self._create_chunk_doc(text, metadata)]

        # 按句子分割
        sentences = self._split_into_sentences(text)

        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(self._create_chunk_doc(current, metadata))
                # 如果单个句子就超出限制，直接截断
                if len(sent) > self.chunk_size:
                    chunks.append(self._create_chunk_doc(sent[:self.chunk_size], metadata))
                    sent = sent[self.chunk_size:]
                    # 继续处理剩余部分
                    while len(sent) > self.chunk_size:
                        chunks.append(self._create_chunk_doc(sent[:self.chunk_size], metadata))
                        sent = sent[self.chunk_size:]
                    current = sent if len(sent) >= self.min_chunk_length else ""
                else:
                    current = sent if len(sent) >= self.min_chunk_length else ""

        if current:
            chunks.append(self._create_chunk_doc(current, metadata))

        return chunks

    def _create_chunk_doc(self, content: str, metadata: dict) -> Document:
        """创建单个chunk Document"""
        new_metadata = metadata.copy()
        new_metadata['original_length'] = len(content)
        return Document(page_content=content, metadata=new_metadata)

    def _detect_chunk_type(self, content: str) -> str:
        """检测chunk类型"""
        content_lower = content.lower()

        # 表格检测
        if '|' in content or '表格' in content or '列名' in content:
            return 'table'

        # 列表检测
        if re.match(r'^\d+[、.]', content, re.MULTILINE):
            return 'list'

        # 标题检测
        if re.match(r'^#{1,6}\s+', content) or re.match(r'^\d+[.:\s]', content):
            return 'heading'

        # 段落检测
        if '\n' in content:
            return 'paragraph'

        return 'sentence'

    def split_text(self, text: str) -> List[str]:
        """分割单个文本"""
        paragraphs = self._split_by_paragraphs(text)
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.chunk_size:
                if current:
                    current += "\n\n"
                current += para
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks


class TextSplitter:
    """兼容旧接口的分块器"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_splitter = SemanticTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        if not documents:
            return []

        chunks = self.semantic_splitter.split_documents(documents)
        print(f"将 {len(documents)} 个文档分割为 {len(chunks)} 个块")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """分割单个文本"""
        return self.semantic_splitter.split_text(text)
