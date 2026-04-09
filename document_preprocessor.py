"""
文档预处理器 - 预处理阶段：拆分 -> 清理 -> 标准化 -> 标签生成 + 图片处理
"""

import re
from pathlib import Path
from typing import List, Optional, Set, Dict
from langchain_core.documents import Document

# 图片处理模块
try:
    from image_processor import ImagePreprocessor
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSOR_AVAILABLE = False
    ImagePreprocessor = None


class DocumentPreprocessor:
    """文档预处理器"""

    # 常见页眉页脚模式
    HEADER_FOOTER_PATTERNS = [
        r'^第\s*\d+\s*页\s*$',  # 第 1 页
        r'^\d+\s*$',  # 纯数字页码
        r'^\s*Page\s+\d+\s*$',  # Page 1
        r'^第\s*[一二三四五六七八九十百千]+\s*页\s*$',  # 中文数字页码
    ]

    # 水印模式
    WATERMARK_PATTERNS = [
        r'^(内部资料|机密|绝密|草稿|未生效|DRAFT)$',
        r'^\[?\s*(内部|机密|秘密)\s*\]?$',
    ]

    # Word文档标题模式（罗马数字、阿拉伯数字、中文章节等）
    WORD_HEADING_PATTERNS = [
        # 阿拉伯数字编号: 1.  1.1  1.1.1
        r'^(\d+(?:\.\d+)*)\s+([^\d].*)$',
        # 罗马数字: I.  II.  III.
        r'^([IVXLC]+)\.\s+(.*)$',
        # 中文章节: 第一章  第一节  第一条
        r'^第([一二三四五六七八九十百千\d]+)\s*[章篇节条]\s+(.*)$',
        # 带括号的编号: (1)  (一)
        r'^(\([一二三四五六七八九十\d]+\))\s+(.*)$',
    ]

    # ========== 新增：文档元数据表格区域模式 ==========
    METADATA_TABLE_PATTERNS = [
        r'^文档编号\s*',           # 文档编号
        r'^项目名称\s*',           # 项目名称
        r'^需求名称\s*',           # 需求名称
        r'^程\s*序\s*名\s*',      # 程序名（可能有空格）
        r'^事务代码\s*',           # 事务代码
        r'^关键用户\s*',           # 关键用户
        r'^用\s*户\s*',           # 用户
        r'^业务顾问\s*',           # 业务顾问
        r'^版\s*本\s*',           # 版本
        r'^分\s*类\s*',           # 分类
        r'^需求类型\s*',           # 需求类型
        r'^开发类型\s*',           # 开发类型
        r'^运行频率\s*',           # 运行频率
        r'^优先级别\s*',           # 优先级别
        r'^基本内容\s*',           # 基本内容
        r'^基本\s*内容\s*',       # 基本内容（可能有空格）
    ]

    # ========== 新增：目录区域模式 ==========
    TABLE_OF_CONTENTS_PATTERNS = [
        r'^目\s*录$',              # 目 录（可能有空格）
        r'^目\s*录\s',            # 目录开头
        r'^TOC$',                  # TOC
        r'^Table of Contents',    # 英文目录
    ]

    # ========== 新增：审批和签名区域模式 ==========
    APPROVAL_SIGNATURE_PATTERNS = [
        r'^审\s*批\s*内\s*容',   # 审批内容
        r'^修改历史',              # 修改历史
        r'^签名\s*',              # 签名
        r'^签名日期\s*',           # 签名日期
        r'^项目\s*公司\s*姓名',   # 审批表格表头
        r'^功能规范确认\s*',       # 功能规范确认
        r'^供应商\s*',            # 供应商
        r'^程序测试确认\s*',       # 程序测试确认
    ]

    # ========== 新增：表格分隔线模式 ==========
    TABLE_SEPARATOR_PATTERNS = [
        r'^[－\-=]{3,}$',         # -----  或 ======
        r'^\s*[｜|]\s*$',         # 表格竖线 |
        r'^[\s│┆├─┣┤┴┬┼]+\s*$',  # 各种表格线
    ]

    # ========== 新增：签字签名区域模式 ==========
    SIGNATURE_LINE_PATTERNS = [
        r'^功能规范确认',          # 功能规范确认行
        r'^供应商',                # 供应商行
        r'^程序测试确认',          # 程序测试确认
        r'^\s*(项目经理|业务部负责人|实施顾问)\s*$',  # 签名人
    ]

    def __init__(self, enable_llm: bool = False, llm_provider=None, enable_image_processing: bool = False, vision_provider: str = None):
        """
        初始化预处理器

        Args:
            enable_llm: 是否启用LLM标签生成
            llm_provider: LLM提供商实例
            enable_image_processing: 是否启用图片处理
            vision_provider: 视觉模型提供商: "ollama" | "minimax"（可选，默认从config读取）
        """
        self.enable_llm = enable_llm
        self.llm_provider = llm_provider
        self.enable_image_processing = enable_image_processing and IMAGE_PROCESSOR_AVAILABLE
        self._seen_hashes: Set[str] = set()

        # 初始化图片处理器
        # 视觉描述依赖于 llm_provider 是否可用，而不是 enable_llm
        if self.enable_image_processing:
            self.image_processor = ImagePreprocessor(
                llm_provider=llm_provider,
                enable_vision=llm_provider is not None,  # 只要有 LLM 提供商就启用视觉
                vision_provider=vision_provider
            )
        else:
            self.image_processor = None

    def preprocess(self, documents: List[Document]) -> List[Document]:
        """
        预处理流程：图片处理 -> 拆分 -> 清理 -> 标准化 -> 标签生成

        Args:
            documents: 原始文档列表

        Returns:
            预处理后的文档列表
        """
        if not documents:
            return documents

        print(f"开始预处理 {len(documents)} 个文档...")

        try:
            # 步骤0：图片处理（提取图片并生成描述）
            if self.image_processor:
                documents = self._process_images_in_documents(documents)
                print(f"  图片处理后: {len(documents)} 个文档")

            # 步骤1：自动拆分（多级）
            split_docs = self.split_by_structure(documents)
            print(f"  拆分后: {len(split_docs)} 个文档")

            # 步骤2：冗余清理
            cleaned_docs = self.clean_redundancy(split_docs)
            print(f"  清理后: {len(cleaned_docs)} 个文档")

            # 步骤3：格式标准化
            normalized_docs = self.normalize_format(cleaned_docs)
            print(f"  标准化后: {len(normalized_docs)} 个文档")

            # 步骤4：标签生成
            if self.enable_llm and self.llm_provider:
                tagged_docs = self.generate_tags_llm(normalized_docs)
            else:
                tagged_docs = self.generate_tags(normalized_docs)
            print(f"  标签生成后: {len(tagged_docs)} 个文档")

            print("预处理完成")
            return tagged_docs

        except Exception as e:
            print(f"预处理失败: {e}，回退到原始文档")
            return documents

    def _process_images_in_documents(self, documents: List[Document]) -> List[Document]:
        """处理文档中的图片，提取描述并插入内容"""
        # 按源文件分组处理
        files_processed = {}  # {file_path: [doc_indices]}

        for idx, doc in enumerate(documents):
            source = doc.metadata.get('source', '')
            if source:
                if source not in files_processed:
                    files_processed[source] = []
                files_processed[source].append(idx)

        # 处理每个文件
        for file_path, doc_indices in files_processed.items():
            if not file_path:
                continue

            # 检查是否为可处理的文件类型
            ext = file_path.lower()
            if not any(ext.endswith(e) for e in ['.pdf', '.docx', '.doc']):
                continue

            print(f"  处理图片: {Path(file_path).name}")

            try:
                # 提取图片并生成描述
                images = self.image_processor.process_document(file_path)

                if images:
                    # 构建图片描述文本
                    image_descriptions = []
                    for img_info in images:
                        desc = img_info.get('description', '')
                        page = img_info.get('page', img_info.get('index', 0) + 1)
                        image_descriptions.append(f"[图片{page}: {desc}]")

                    combined_desc = "\n\n".join(image_descriptions)

                    # 将描述插入到相关文档中
                    for idx in doc_indices:
                        doc = documents[idx]
                        original_content = doc.page_content

                        # 在文档末尾添加图片描述
                        new_content = f"{original_content}\n\n---\n图片内容:\n{combined_desc}"

                        new_metadata = doc.metadata.copy()
                        new_metadata['has_images'] = True
                        new_metadata['image_count'] = len(images)

                        documents[idx] = Document(
                            page_content=new_content,
                            metadata=new_metadata
                        )

            except Exception as e:
                print(f"  图片处理失败 {file_path}: {e}")

        return documents

    def split_by_structure(self, documents: List[Document]) -> List[Document]:
        """按文档结构拆分（章节、段落）"""
        result = []

        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()

            # 记录原始来源
            metadata['original_source'] = metadata.get('source', '')

            # 检测文档类型
            source_path = metadata.get('source', '')
            is_word_doc = any(source_path.lower().endswith(ext) for ext in ['.docx', '.doc'])

            # 按文档类型选择拆分策略
            if '#' in content:
                # Markdown文档
                split_docs = self._split_by_markdown_headings(content, metadata)
                result.extend(split_docs)
            elif is_word_doc or self._has_word_headings(content):
                # Word文档
                split_docs = self._split_by_word_headings(content, metadata)
                result.extend(split_docs)
            else:
                # 无标题则按段落拆分
                split_docs = self._split_by_paragraphs(content, metadata)
                result.extend(split_docs)

        # 合并过短的多个相邻文档
        result = self._merge_short_docs(result)

        return result

    def _has_word_headings(self, content: str) -> bool:
        """检测内容是否包含Word风格标题"""
        lines = content.split('\n')
        heading_count = 0

        for line in lines[:20]:  # 只检查前20行
            line = line.strip()
            if not line:
                continue

            for pattern in self.WORD_HEADING_PATTERNS:
                if re.match(pattern, line):
                    heading_count += 1
                    if heading_count >= 2:  # 至少2个标题才认为是Word文档
                        return True
                    break

        return False

    def _split_by_word_headings(self, content: str, metadata: dict) -> List[Document]:
        """按Word标题样式拆分"""
        lines = content.split('\n')
        sections = []
        current_section = {'level': 0, 'title': '', 'content': []}

        for line in lines:
            line_stripped = line.strip()
            is_heading = False

            # 尝试匹配Word标题模式
            for pattern in self.WORD_HEADING_PATTERNS:
                match = re.match(pattern, line_stripped)
                if match:
                    # 保存前一个section
                    if current_section['content']:
                        sections.append(current_section)

                    # 确定标题层级
                    level = self._get_word_heading_level(match.group(1))
                    title = match.group(2).strip() if match.lastindex >= 2 else match.group(1)

                    current_section = {
                        'level': level,
                        'title': title,
                        'content': [line]
                    }
                    is_heading = True
                    break

            if not is_heading:
                current_section['content'].append(line)

        # 保存最后一个section
        if current_section['content']:
            sections.append(current_section)

        # 转换为Document列表
        result = []
        for i, section in enumerate(sections):
            section_content = '\n'.join(section['content'])

            # 跳过纯标题或空内容
            if not section_content.strip() or section_content.strip() == section['title']:
                continue

            section_metadata = metadata.copy()
            section_metadata['section_level'] = section['level']
            section_metadata['section_title'] = section['title']
            section_metadata['section_index'] = i
            section_metadata['doc_type'] = 'word'

            doc = Document(
                page_content=section_content,
                metadata=section_metadata
            )
            result.append(doc)

        return result

    def _get_word_heading_level(self, heading_num: str) -> int:
        """根据标题编号确定层级"""
        # 数字编号: 1 -> 1, 1.1 -> 2, 1.1.1 -> 3
        if re.match(r'^\d+(\.\d+)*$', heading_num):
            dots = heading_num.count('.')
            return dots + 1

        # 罗马数字
        if re.match(r'^[IVXLC]+$', heading_num.upper()):
            return 1

        # 中文数字
        if re.search(r'[一二三四五六七八九十百千]', heading_num):
            return 1

        # 带括号的编号
        return 2

    def _split_by_markdown_headings(self, content: str, metadata: dict) -> List[Document]:
        """按Markdown标题拆分"""
        # 匹配Markdown标题 (# ## ### 等)
        heading_pattern = r'^(#{1,6})\s+(.+)$'

        lines = content.split('\n')
        sections = []
        current_section = {'level': 0, 'title': '', 'content': []}

        for line in lines:
            match = re.match(heading_pattern, line.strip())
            if match:
                # 保存前一个section
                if current_section['content']:
                    sections.append(current_section)

                # 开始新的section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': [line]
                }
            else:
                current_section['content'].append(line)

        # 保存最后一个section
        if current_section['content']:
            sections.append(current_section)

        # 转换为Document列表
        result = []
        for i, section in enumerate(sections):
            section_content = '\n'.join(section['content'])

            # 跳过纯标题或空内容
            if not section_content.strip() or section_content.strip() == section['title']:
                continue

            section_metadata = metadata.copy()
            section_metadata['section_level'] = section['level']
            section_metadata['section_title'] = section['title']
            section_metadata['section_index'] = i

            doc = Document(
                page_content=section_content,
                metadata=section_metadata
            )
            result.append(doc)

        return result

    def _split_by_paragraphs(self, content: str, metadata: dict) -> List[Document]:
        """按段落拆分"""
        # 按双换行或单换行拆分
        paragraphs = re.split(r'\n\s*\n|\n', content)

        result = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            para_metadata = metadata.copy()
            para_metadata['paragraph_index'] = i

            doc = Document(
                page_content=para,
                metadata=para_metadata
            )
            result.append(doc)

        return result

    def _merge_short_docs(self, documents: List[Document], min_length: int = 50) -> List[Document]:
        """合并过短的多个相邻文档"""
        if not documents:
            return documents

        result = [documents[0]]

        for doc in documents[1:]:
            # 如果当前文档太短，尝试与前一个合并
            if len(doc.page_content) < min_length and result:
                last_doc = result[-1]
                merged_content = last_doc.page_content + '\n\n' + doc.page_content

                merged_metadata = last_doc.metadata.copy()
                merged_metadata.update(doc.metadata)

                result[-1] = Document(
                    page_content=merged_content,
                    metadata=merged_metadata
                )
            else:
                result.append(doc)

        return result

    def clean_redundancy(self, documents: List[Document]) -> List[Document]:
        """清理冗余内容"""
        self._seen_hashes.clear()
        result = []

        for doc in documents:
            content = doc.page_content

            # 1. 清理页眉页脚和页码
            content = self._clean_headers_footers(content)

            # 2. 清理水印
            content = self._clean_watermarks(content)

            # 3. 清理文档元数据表格区域（基本信息表格）
            content = self._clean_metadata_table(content)

            # 4. 清理目录区域
            content = self._clean_table_of_contents(content)

            # 5. 清理审批和签名区域
            content = self._clean_approval_signature(content)

            # 6. 清理表格分隔线
            content = self._clean_table_separators(content)

            # 7. 去除重复内容（基于内容哈希）
            content_hash = self._compute_content_hash(content)
            if content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(content_hash)

            # 8. 跳过过于简短的文档
            if len(content.strip()) < 20:
                continue

            # 更新metadata
            new_metadata = doc.metadata.copy()
            new_metadata['preprocessed'] = True

            result.append(Document(
                page_content=content,
                metadata=new_metadata
            ))

        return result

    def _clean_metadata_table(self, content: str) -> str:
        """清理文档元数据表格区域（基本信息表格）"""
        lines = content.split('\n')
        cleaned_lines = []
        skip_mode = False
        blank_line_count = 0

        for line in lines:
            line_stripped = line.strip()

            # 检查是否匹配元数据表格模式
            is_metadata = False
            for pattern in self.METADATA_TABLE_PATTERNS:
                if re.match(pattern, line_stripped):
                    is_metadata = True
                    break

            if is_metadata:
                skip_mode = True
                blank_line_count = 0
                continue

            # 如果处于跳过模式，检测是否应该退出
            if skip_mode:
                # 遇到空行计数，连续2个以上空行认为元数据区域结束
                if not line_stripped:
                    blank_line_count += 1
                    if blank_line_count >= 2:
                        skip_mode = False
                        blank_line_count = 0
                        cleaned_lines.append(line)
                # 检测是否到达真正的章节标题
                elif self._is_real_heading(line_stripped):
                    skip_mode = False
                    cleaned_lines.append(line)
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _clean_table_of_contents(self, content: str) -> str:
        """清理目录区域"""
        lines = content.split('\n')
        cleaned_lines = []
        in_toc = False
        toc_line_count = 0

        for line in lines:
            line_stripped = line.strip()

            # 检测目录开始
            if not in_toc:
                for pattern in self.TABLE_OF_CONTENTS_PATTERNS:
                    if re.match(pattern, line_stripped):
                        in_toc = True
                        toc_line_count = 0
                        break

                if in_toc:
                    continue  # 跳过目录标题行本身

            if in_toc:
                toc_line_count += 1

                # 目录条目通常是数字编号开头
                is_toc_entry = bool(re.match(r'^\d+[\.. ]', line_stripped)) or \
                               bool(re.match(r'^第.*章', line_stripped)) or \
                               bool(re.match(r'^第.*节', line_stripped))

                # 目录通常不超过30行，或者遇到非目录条目时停止
                if toc_line_count > 50 or (line_stripped and not is_toc_entry and not re.match(r'^\s*$', line_stripped)):
                    in_toc = False
                    cleaned_lines.append(line)
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _clean_approval_signature(self, content: str) -> str:
        """清理审批和签名区域"""
        lines = content.split('\n')
        cleaned_lines = []
        skip_mode = False
        skip_start_pattern = None

        for line in lines:
            line_stripped = line.strip()

            # 检测审批/签名区域开始
            if re.match(r'^审\s*批\s*内\s*容', line_stripped) or \
               re.match(r'^修改历史', line_stripped) or \
               re.match(r'^签名\s*', line_stripped) or \
               re.match(r'^功能规范确认', line_stripped) or \
               re.match(r'^供应商', line_stripped) or \
               re.match(r'^程序测试确认', line_stripped):
                skip_mode = True
                skip_start_pattern = line_stripped
                continue

            if skip_mode:
                # 检查是否到达文档末尾或下一个真正的章节
                if self._is_real_heading(line_stripped):
                    skip_mode = False
                    cleaned_lines.append(line)
                # 如果跳过了太多行（超过20行），强制退出跳过模式
                elif len(cleaned_lines) > 0 and not line_stripped:
                    # 遇到空行认为审批区域结束
                    continue
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _clean_table_separators(self, content: str) -> str:
        """清理表格分隔线，但保护Markdown表格"""
        lines = content.split('\n')
        cleaned_lines = []
        in_markdown_table = False

        for line in lines:
            line_stripped = line.strip()

            # 检测Markdown表格上下文
            if line_stripped.startswith('|') and line_stripped.endswith('|'):
                in_markdown_table = True
                cleaned_lines.append(line)
                continue

            # 如果上一行是Markdown表格，当前行是分隔线则保留
            if in_markdown_table and (line_stripped.startswith('|') or '---' in line_stripped):
                cleaned_lines.append(line)
                if '---' not in line_stripped and '|' not in line_stripped:
                    in_markdown_table = False
                continue

            # 重置Markdown表格标记
            if in_markdown_table and line_stripped and not line_stripped.startswith('|'):
                in_markdown_table = False

            # 检查是否匹配表格分隔线模式
            is_separator = False
            for pattern in self.TABLE_SEPARATOR_PATTERNS:
                if re.match(pattern, line_stripped):
                    is_separator = True
                    break

            if not is_separator:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _is_real_heading(self, line: str) -> bool:
        """判断是否是真标题（而非表格内容）"""
        if not line:
            return False

        # 章节标题模式
        heading_patterns = [
            r'^\d+\.\d+\s+',        # 1.1  4.1.2
            r'^\d+\s+[^\d\s]',      # 1. xxx  (数字+空格+非数字开头)
            r'^第[一二三四五六七八九十百千\d]+[章篇节条]',  # 第一章
            r'^#{1,6}\s+',         # Markdown标题
            r'^\([一二三四五六七八九十\d]+\)\s+',  # (1) 标题
            r'^[IVX]+\.\s+',        # I. II. 罗马数字
        ]

        for pattern in heading_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _clean_headers_footers(self, content: str) -> str:
        """清理页眉页脚和页码"""
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            # 跳过匹配页眉页脚模式的行
            is_header_footer = False
            for pattern in self.HEADER_FOOTER_PATTERNS:
                if re.match(pattern, line.strip()):
                    is_header_footer = True
                    break

            if not is_header_footer:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _clean_watermarks(self, content: str) -> str:
        """清理水印"""
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            # 跳过匹配水印模式的行
            is_watermark = False
            for pattern in self.WATERMARK_PATTERNS:
                if re.match(pattern, line.strip()):
                    is_watermark = True
                    break

            if not is_watermark:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _compute_content_hash(self, content: str) -> str:
        """计算内容哈希（用于去重）"""
        # 归一化内容后计算哈希
        normalized = self._normalize_text(content)
        return str(hash(normalized))

    def _normalize_text(self, text: str) -> str:
        """归一化文本用于比较"""
        # 转为小写，去除空白
        text = text.lower()
        text = re.sub(r'\s+', '', text)
        return text

    def normalize_format(self, documents: List[Document]) -> List[Document]:
        """格式标准化"""
        result = []

        for doc in documents:
            content = doc.page_content

            # 1. 统一换行符
            content = content.replace('\r\n', '\n').replace('\r', '\n')

            # 2. 合并多余的空行（超过2个连续空行改为2个）
            content = re.sub(r'\n{3,}', '\n\n', content)

            # 3. 去除行首行尾空白
            lines = [line.strip() for line in content.split('\n')]
            content = '\n'.join(lines)

            # 4. 去除不可见字符（保留中文和常用字符）
            content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', content)

            # 5. 规范化全角标点为半角（可选，默认不启用）
            # content = self._normalize_punctuation(content)

            result.append(Document(
                page_content=content,
                metadata=doc.metadata
            ))

        return result

    def generate_tags(self, documents: List[Document]) -> List[Document]:
        """基于规则生成标签"""
        result = []

        for doc in documents:
            tags = set()
            content = doc.page_content
            metadata = doc.metadata

            # ========== 新增：添加文档名+版本号标签 ==========
            doc_name_version = self._extract_doc_name_version(metadata)
            if doc_name_version:
                tags.add(doc_name_version)

            # 1. 从标题提取标签
            if 'section_title' in metadata:
                title = metadata['section_title']
                # 提取标题中的关键词作为标签
                title_tags = self._extract_keywords_from_title(title)
                tags.update(title_tags)

            # 2. 从开头内容提取标签
            first_lines = content.split('\n')[:5]
            intro_text = ' '.join(first_lines)
            intro_tags = self._extract_keywords_from_text(intro_text, max_tags=3)
            tags.update(intro_tags)

            # 3. 基于章节层级添加标签
            if 'section_level' in metadata:
                level = metadata['section_level']
                if level == 1:
                    tags.add('chapter')
                elif level == 2:
                    tags.add('section')
                elif level == 3:
                    tags.add('subsection')

            # 更新metadata
            new_metadata = metadata.copy()
            new_metadata['tags'] = list(tags) if tags else []

            result.append(Document(
                page_content=content,
                metadata=new_metadata
            ))

        return result

    def _extract_doc_name_version(self, metadata: dict) -> str:
        """从metadata中提取文档名+版本号"""
        # 获取源文件路径
        source = metadata.get('source', '')
        version = metadata.get('version', '')

        if not source:
            return ''

        # 从文件路径中提取文件名（不含扩展名）
        from pathlib import Path
        try:
            doc_name = Path(source).stem  # 文件名（不含扩展名）
        except Exception:
            doc_name = source

        # 如果有版本号，拼接为 "文档名_V版本"
        if version:
            # 清理版本号格式
            version_clean = version.replace('/', '-').replace(':', '-')
            return f"{doc_name}_V{version_clean}"
        else:
            return doc_name

    def _extract_keywords_from_title(self, title: str) -> List[str]:
        """从标题提取关键词"""
        # 去除Markdown标题符号
        title = re.sub(r'^#+\s*', '', title)

        # 常见章节词
        section_words = ['概述', '简介', '介绍', '定义', '说明', '要求', '规范', '标准', '流程', '步骤', '方法', '方案', '总结', '结论']

        tags = []
        for word in section_words:
            if word in title:
                tags.append(word)

        return tags

    def _extract_keywords_from_text(self, text: str, max_tags: int = 5) -> List[str]:
        """从文本提取关键词"""
        # 简化实现：提取前几个词作为标签
        # 实际生产中可使用TF-IDF、TextRank等算法

        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        words = text.split()

        # 过滤掉过短的词
        tags = [w for w in words if len(w) >= 2][:max_tags]

        return tags

    def generate_tags_llm(self, documents: List[Document]) -> List[Document]:
        """使用LLM生成语义标签（仅对重要章节）"""
        if not self.llm_provider:
            return self.generate_tags(documents)

        result = []

        for i, doc in enumerate(documents):
            content = doc.page_content
            metadata = doc.metadata

            # ========== 新增：添加文档名+版本号标签 ==========
            doc_name_version = self._extract_doc_name_version(metadata)
            tags = set()
            if doc_name_version:
                tags.add(doc_name_version)

            # 只对较长或有标题的文档使用LLM生成标签
            use_llm = (
                len(content) > 500 or
                ('section_title' in metadata and metadata['section_title'])
            )

            if use_llm and (i % 3 == 0):  # 每3个文档处理一次，避免耗时过长
                try:
                    llm_tags = self._generate_tags_with_llm(content, metadata)
                    tags.update(llm_tags)
                except Exception as e:
                    print(f"  LLM标签生成失败: {e}")

            # 合并规则提取的标签
            rule_tags = set(self._extract_keywords_from_title(
                metadata.get('section_title', '')
            ))
            tags.update(rule_tags)

            # 更新metadata
            new_metadata = metadata.copy()
            new_metadata['tags'] = list(tags) if tags else []

            result.append(Document(
                page_content=content,
                metadata=new_metadata
            ))

        return result

    def _generate_tags_with_llm(self, content: str, metadata: dict) -> set:
        """使用LLM生成标签"""
        # 截取内容前500个字符
        preview = content[:500]

        prompt = f"""请为以下文档内容生成3-5个关键词标签（标签应该是名词或短语）。

文档内容预览：
{preview}

请只返回标签，用逗号分隔，不要有其他内容。"""

        try:
            response = self.llm_provider.chat(prompt)
            # 解析标签
            tags_str = response.strip()
            # 去除可能的标点
            tags = [t.strip().strip('.,;:!?') for t in tags_str.split(',')]
            tags = [t for t in tags if t]  # 过滤空字符串
            return set(tags[:5])  # 最多5个标签
        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {e}")
