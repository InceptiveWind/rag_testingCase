"""
文本分块器 - 支持多级语义分块
"""

import re
from typing import List, Set, Tuple
from langchain_core.documents import Document


class SemanticTextSplitter:
    """语义分块器 - 按句子/段落分块，保持语义完整性"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 50, min_chunk_length: int = 50):
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

        # 检测是否为表格文档，启用表格保护
        is_table_doc = metadata.get('has_tables', False) or self._is_markdown_table_content(content)

        # 第一级：按段落分割
        if is_table_doc:
            # 表格文档按表格分割，保持表格完整
            segments = self._split_by_tables_or_paragraphs(content)
        else:
            segments = self._split_by_paragraphs(content)

        chunks = []
        current_chunk = ""
        current_segment_is_table = False
        # 当前重叠内容（用于下一个chunk）
        current_overlap = ""

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            is_table = self._is_markdown_table(seg)
            is_test_case_structure = is_test_case and self._is_test_case_key_structure(seg)

            # 如果是新的大段（段落或表格分隔）
            if current_chunk and (is_table != current_segment_is_table or
                                  len(current_chunk) + len(seg) + 2 > self.chunk_size):
                # 保存当前chunk
                if current_chunk.strip():
                    chunk_docs = self._create_chunks(current_chunk, metadata, current_overlap)
                    chunks.extend(chunk_docs)
                    # 更新重叠内容
                    if chunk_docs:
                        last_chunk = chunk_docs[-1].page_content
                        current_overlap = self._calculate_semantic_overlap(last_chunk)
                        print(f"[DEBUG split_document] 更新current_overlap (段落分割), 长度={len(current_overlap)}")
                current_chunk = ""
                current_overlap = ""  # 重置重叠，因为当前chunk已处理

            # 如果是测试用例关键结构，检查是否需要保持完整
            if is_test_case_structure:
                if current_chunk:
                    chunk_docs = self._create_chunks(current_chunk, metadata, current_overlap)
                    chunks.extend(chunk_docs)
                    if chunk_docs:
                        last_chunk = chunk_docs[-1].page_content
                        current_overlap = self._calculate_semantic_overlap(last_chunk)
                        print(f"[DEBUG split_document] 更新current_overlap (测试用例前), 长度={len(current_overlap)}")
                    current_chunk = ""

                # 判断是否保持重叠
                if not self._should_keep_overlap(seg, 'test_case'):
                    current_overlap = ""
                    print(f"[DEBUG split_document] 重置current_overlap (测试用例关键结构)")

                if len(seg) > self.chunk_size * 1.5:
                    # 使用结构保持分割
                    test_chunks = self._split_with_structure_preserved(seg, metadata)
                    chunks.extend(test_chunks)
                    if test_chunks:
                        last_chunk = test_chunks[-1].page_content
                        current_overlap = self._calculate_semantic_overlap(last_chunk)
                else:
                    chunks.append(self._create_chunk_doc(seg, metadata))
                    current_overlap = self._calculate_semantic_overlap(seg)
                current_segment_is_table = False
            elif is_table:
                # 表格：保持完整，不与其他内容混合
                if current_chunk:
                    chunk_docs = self._create_chunks(current_chunk, metadata, current_overlap)
                    chunks.extend(chunk_docs)
                    if chunk_docs:
                        last_chunk = chunk_docs[-1].page_content
                        current_overlap = self._calculate_semantic_overlap(last_chunk)
                        print(f"[DEBUG split_document] 更新current_overlap (表格前), 长度={len(current_overlap)}")
                    current_chunk = ""

                # 判断是否保持重叠
                if not self._should_keep_overlap(seg, 'table'):
                    current_overlap = ""
                    print(f"[DEBUG split_document] 重置current_overlap (表格)")

                # 检查表格是否超长
                if len(seg) > self.chunk_size:
                    # 按行分割表格但保持结构
                    table_chunks = self._split_table_preserve(seg, metadata)
                    chunks.extend(table_chunks)
                    if table_chunks:
                        last_chunk = table_chunks[-1].page_content
                        current_overlap = self._calculate_semantic_overlap(last_chunk)
                else:
                    chunks.append(self._create_chunk_doc(seg, metadata))
                    current_overlap = self._calculate_semantic_overlap(seg)
                current_segment_is_table = False
            else:
                # 普通段落：累积到current_chunk
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += seg
                current_segment_is_table = False

        # 处理最后一个chunk
        if current_chunk:
            chunks.extend(self._create_chunks(current_chunk, metadata, current_overlap))

        # 标记chunk类型
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_type'] = self._detect_chunk_type(chunk.page_content)

        return chunks

    def _is_markdown_table_content(self, content: str) -> bool:
        """检测内容是否包含Markdown表格"""
        lines = content.split('\n')
        table_lines = 0
        for line in lines:
            if self._is_markdown_table(line):
                table_lines += 1
                if table_lines >= 1:
                    return True
        return False

    def _is_markdown_table(self, text: str) -> bool:
        """检测单行是否为Markdown表格行"""
        text = text.strip()
        if not text:
            return False
        # Markdown表格格式: | xxx | xxx | xxx |
        if text.startswith('|') and text.endswith('|'):
            parts = text.split('|')
            # 至少应该有 2 个 | 分隔符（3列以上）
            if len(parts) >= 4:
                return True
        return False

    def _split_by_tables_or_paragraphs(self, content: str) -> List[str]:
        """按表格或段落分割（保持表格完整）"""
        lines = content.split('\n')
        segments = []
        current_paragraph_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            is_table_line = self._is_markdown_table(line)

            if is_table_line:
                # 保存之前的非表格段落
                if current_paragraph_lines:
                    para_text = '\n'.join(current_paragraph_lines)
                    if para_text.strip():
                        segments.append(para_text)
                    current_paragraph_lines = []

                # 收集完整的表格（表格行 + 分隔行 + 数据行，直到遇到空行或新的段落内容）
                table_lines = [line]
                i += 1

                # 收集表格的后续行
                while i < len(lines):
                    next_line = lines[i]
                    next_is_table = self._is_markdown_table(next_line)

                    if next_is_table:
                        # 表格行，加入表格
                        table_lines.append(next_line)
                        i += 1
                    elif '---' in next_line or '|' in next_line[:5]:
                        # 可能是表格分隔行或被截断的表格行
                        stripped = next_line.strip()
                        if stripped.startswith('|') or stripped == '' or stripped.startswith('---'):
                            if stripped.startswith('|') or stripped.startswith('---'):
                                table_lines.append(next_line)
                                i += 1
                            else:
                                # 空行或分隔行
                                table_lines.append(next_line)
                                i += 1
                        else:
                            # 非表格内容，结束表格
                            break
                    elif next_line.strip() == '':
                        # 空行，表格结束
                        table_lines.append(next_line)
                        i += 1
                        break
                    else:
                        # 非表格段落内容，表格结束
                        break

                # 保存完整的表格
                table_text = '\n'.join(table_lines)
                if table_text.strip():
                    segments.append(table_text)
            else:
                # 非表格行
                current_paragraph_lines.append(line)
                i += 1

        # 保存最后一个段落
        if current_paragraph_lines:
            para_text = '\n'.join(current_paragraph_lines)
            if para_text.strip():
                segments.append(para_text)

        return segments

    def _split_table_preserve(self, table_content: str, metadata: dict) -> List[Document]:
        """保持表格结构地分割（用于超长表格）"""
        lines = table_content.split('\n')
        if len(lines) <= 3:
            # 只有表头和分隔行，直接返回
            return [self._create_chunk_doc(table_content, metadata)]

        chunks = []
        current_chunk_lines = []

        # 保留表头和分隔行
        header_lines = lines[:2]  # 表头 + 分隔行

        for line in lines[2:]:  # 从数据行开始
            test_chunk = '\n'.join(current_chunk_lines + [line])

            if len(test_chunk) <= self.chunk_size:
                current_chunk_lines.append(line)
            else:
                # 当前块满了，保存并开始新块
                if current_chunk_lines:
                    # 新块也需要表头
                    chunk_content = '\n'.join(header_lines + current_chunk_lines)
                    chunks.append(self._create_chunk_doc(chunk_content, metadata))
                current_chunk_lines = [line]

        # 处理最后一块
        if current_chunk_lines:
            chunk_content = '\n'.join(header_lines + current_chunk_lines)
            chunks.append(self._create_chunk_doc(chunk_content, metadata))

        # 如果产生多块，标记为被分割的表格
        for chunk in chunks:
            if len(chunks) > 1:
                chunk.metadata['table_split'] = True
                chunk.metadata['table_split_index'] = chunks.index(chunk)
                chunk.metadata['table_split_total'] = len(chunks)

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

    def _split_into_sub_clauses(self, text: str) -> List[str]:
        """将超长句子按子句分割（逗号、顿号、分号）"""
        sub_clauses = []
        current = ""
        # 子句分隔符：中文逗号、顿号、分号 + 英文逗号、分号
        sub_delimiters = '，、；,;'

        for char in text:
            current += char
            if char in sub_delimiters:
                sub_clauses.append(current)
                current = ""

        if current.strip():
            sub_clauses.append(current)

        return sub_clauses if sub_clauses else [text]

    def _find_semantic_boundary(self, text: str, start_pos: int, direction: str = 'backward') -> int:
        """查找最近的语义边界位置

        Args:
            text: 文本内容
            start_pos: 起始位置（字符索引）
            direction: 'backward'向后查找，'forward'向前查找

        Returns:
            边界位置索引，找不到则返回-1
        """
        if start_pos < 0 or start_pos > len(text):
            return -1

        # 句子结束符：。！？.!?
        sentence_endings = '。！？.!?'
        # 子句分隔符：，、；,;
        clause_delimiters = '，、；,;'
        # 段落分隔符：换行
        paragraph_delimiters = '\n'

        if direction == 'backward':
            # 向后查找（从start_pos向左）
            # 优先查找句子边界
            for i in range(start_pos - 1, -1, -1):
                char = text[i]
                if char in sentence_endings:
                    # 找到句子边界，返回边界后的位置（i+1）
                    return i + 1
                elif char in clause_delimiters:
                    # 找到子句边界，返回边界后的位置
                    return i + 1
                elif char in paragraph_delimiters:
                    # 找到段落边界
                    return i + 1

            # 没找到边界，返回0（文本开头）
            return 0
        else:
            # 向前查找（从start_pos向右）
            # 优先查找句子边界
            for i in range(start_pos, len(text)):
                char = text[i]
                if char in sentence_endings:
                    # 找到句子边界，返回边界位置（i+1）
                    return i + 1
                elif char in clause_delimiters:
                    # 找到子句边界，返回边界位置
                    return i + 1
                elif char in paragraph_delimiters:
                    # 找到段落边界
                    return i + 1

            # 没找到边界，返回-1
            return -1

    def _calculate_semantic_overlap(self, chunk_content: str) -> str:
        """计算语义完整的重叠内容

        从chunk末尾向前查找，找到最近的语义边界，确保重叠内容完整
        """
        if len(chunk_content) <= self.chunk_overlap:
            # 如果chunk长度小于等于重叠长度，整个chunk作为重叠
            return chunk_content

        # 从目标重叠位置开始查找边界
        target_start = len(chunk_content) - self.chunk_overlap
        boundary_pos = self._find_semantic_boundary(chunk_content, target_start, 'forward')

        if boundary_pos == -1:
            # 向前找不到，则从末尾向后找
            boundary_pos = self._find_semantic_boundary(chunk_content, len(chunk_content), 'backward')

        # 如果找到边界，返回边界之后的内容；否则返回默认重叠
        if boundary_pos != -1:
            return chunk_content[boundary_pos:]
        else:
            # 找不到语义边界，使用简单的字符截取
            return chunk_content[-self.chunk_overlap:]

    def _split_long_clause(self, clause: str, metadata: dict) -> List[Document]:
        """分割超长子句，保持语义边界

        用于处理超长句子分割后的子句，确保分割点在语义边界处
        """
        if len(clause) <= self.chunk_size:
            return [self._create_chunk_doc(clause, metadata)]

        chunks = []
        current = ""

        # 尝试按语义边界分割：先按子句分隔符分割
        sub_parts = []
        temp = ""
        for char in clause:
            temp += char
            if char in '，、；,;。！？.!?':
                sub_parts.append(temp)
                temp = ""
        if temp:
            sub_parts.append(temp)

        # 如果子句分割成功，按子句构建chunks
        if len(sub_parts) > 1:
            for part in sub_parts:
                if len(current) + len(part) <= self.chunk_size:
                    current += part
                else:
                    if current:
                        chunks.append(self._create_chunk_doc(current, metadata))
                    current = part
            if current:
                chunks.append(self._create_chunk_doc(current, metadata))
        else:
            # 无法按子句分割，按字符截断但尽量在单词边界
            # 简单实现：按chunk_size截断
            for i in range(0, len(clause), self.chunk_size):
                chunk_content = clause[i:i + self.chunk_size]
                if chunk_content.strip():
                    chunks.append(self._create_chunk_doc(chunk_content, metadata))

        return chunks

    def _should_keep_overlap(self, segment: str, segment_type: str) -> bool:
        """判断在特殊结构后是否应保持重叠

        Args:
            segment: 当前段内容
            segment_type: 段类型 ('table', 'test_case', 'paragraph')

        Returns:
            True表示应保持重叠，False表示应重置重叠
        """
        if segment_type == 'table':
            # 短小的表格保持重叠
            return len(segment) < self.chunk_size * 0.3
        elif segment_type == 'test_case':
            # 测试用例关键结构通常较短，保持重叠
            return len(segment) < self.chunk_size * 0.5
        else:
            # 普通段落总是保持重叠
            return True

    def _create_chunks(self, text: str, metadata: dict, prefix_overlap: str = "") -> List[Document]:
        """创建chunks，处理重叠和过短问题

        Args:
            text: 要分割的文本
            metadata: chunk元数据
            prefix_overlap: 上一个块的重叠内容，将添加到当前块开头
        """
        if not text.strip():
            return []

        # 检查是否需要分割
        if len(text) <= self.chunk_size:
            if len(text) < self.min_chunk_length:
                return []
            # 如果有前缀重叠，尝试合并
            if prefix_overlap and not text.startswith(prefix_overlap):
                combined = prefix_overlap + text
                if len(combined) <= self.chunk_size:
                    return [self._create_chunk_doc(combined, metadata)]
            return [self._create_chunk_doc(text, metadata)]

        # 按句子分割
        print(f"[DEBUG _create_chunks] 文本长度={len(text)}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        sentences = self._split_into_sentences(text)
        print(f"[DEBUG _create_chunks] 分割出 {len(sentences)} 个句子")

        chunks = []
        current_chunk = prefix_overlap  # 初始化为前缀重叠
        overlap = ""  # 上一个chunk末尾的重叠内容（用于下一个chunk）

        for sent in sentences:
            # 处理超长句子：按子句分割
            if len(sent) > self.chunk_size:
                sub_clauses = self._split_into_sub_clauses(sent)
                for clause in sub_clauses:
                    # 如果有重叠且子句不以重叠开头，则添加重叠
                    if overlap and not clause.startswith(overlap):
                        clause = overlap + clause

                    # 检查子句是否超长
                    if len(clause) > self.chunk_size:
                        # 保存当前chunk
                        if current_chunk.strip():
                            print(f"[DEBUG _create_chunks] 保存当前chunk (超长句子分支1), 长度={len(current_chunk)}")
                            chunks.append(self._create_chunk_doc(current_chunk, metadata))
                            overlap = self._calculate_semantic_overlap(current_chunk)

                        # 超长子句按语义边界分割
                        print(f"[DEBUG _create_chunks] 分割超长子句, 长度={len(clause)}")
                        sub_chunks = self._split_long_clause(clause, metadata)
                        chunks.extend(sub_chunks)
                        if sub_chunks:
                            # 取最后一个子句的重叠作为后续重叠
                            last_sub_chunk = sub_chunks[-1].page_content
                            overlap = self._calculate_semantic_overlap(last_sub_chunk)
                        current_chunk = ""
                    elif len(current_chunk) + len(clause) <= self.chunk_size:
                        # 可以加入当前chunk
                        current_chunk += clause
                    else:
                        # 当前chunk满了，保存并创建新chunk
                        if current_chunk.strip():
                            print(f"[DEBUG _create_chunks] 保存当前chunk (超长句子分支2), 长度={len(current_chunk)}")
                            chunks.append(self._create_chunk_doc(current_chunk, metadata))
                            overlap = self._calculate_semantic_overlap(current_chunk)

                        # 新chunk从当前子句开始（应用重叠）
                        current_chunk = overlap + clause if overlap and not clause.startswith(overlap) else clause
                continue

            # 正常句子：尝试加入当前chunk
            test_chunk = current_chunk + sent
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 当前chunk满了，保存
                if current_chunk.strip():
                    print(f"[DEBUG _create_chunks] 保存当前chunk (正常句子分支), 长度={len(current_chunk)}")
                    chunks.append(self._create_chunk_doc(current_chunk, metadata))
                    overlap = self._calculate_semantic_overlap(current_chunk)

                # 新chunk从当前句子开始（应用重叠）
                current_chunk = overlap + sent if overlap and not sent.startswith(overlap) else sent

        # 处理最后一个chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_length:
            chunks.append(self._create_chunk_doc(current_chunk, metadata))

        return chunks

    def _create_chunk_doc(self, content: str, metadata: dict) -> Document:
        """创建单个chunk Document"""
        new_metadata = metadata.copy()
        new_metadata['original_length'] = len(content)
        return Document(page_content=content, metadata=new_metadata)

    def _detect_chunk_type(self, content: str) -> str:
        """检测chunk类型"""
        content_lower = content.lower()
        content_stripped = content.strip()

        # Markdown表格检测（更精确）
        if content_stripped.startswith('|') and content_stripped.endswith('|'):
            if '---' in content or ' | ' in content:
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

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 50):
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
