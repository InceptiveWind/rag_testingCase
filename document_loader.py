"""
文档加载器 - 支持多种格式的文档加载
"""

import os
import json
import hashlib
import zipfile
from pathlib import Path
from typing import List
from datetime import datetime
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document


def compute_file_hash(file_path: Path) -> str:
    """计算文件的MD5哈希值（基于完整内容）"""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"计算文件哈希失败 {file_path.name}: {e}")
        return ""


class DocumentLoader:
    """文档加载器"""

    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)

    def _add_version_metadata(self, documents: List[Document], file_path: Path) -> List[Document]:
        """为文档添加版本信息到metadata"""
        file_hash = compute_file_hash(file_path)
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        for doc in documents:
            # 更新metadata，添加版本信息
            doc.metadata['file_hash'] = file_hash
            doc.metadata['version'] = version
            # 确保source字段存在
            if 'source' not in doc.metadata:
                doc.metadata['source'] = str(file_path)

        return documents

    def load_file(self, file_path: Path):
        """加载单个文件"""
        suffix = file_path.suffix.lower()

        # 特殊处理：跳过 .ppt (稳定性问题)
        if suffix == '.ppt':
            print(f"跳过(需转换格式): {file_path.name}")
            return None

        # 自定义加载器处理
        if suffix == '.xmind':
            return self._load_xmind(file_path)
        elif suffix == '.vsdx':
            return self._load_vsdx(file_path)
        elif suffix == '.docx':
            return self._load_docx(file_path)
        elif suffix == '.doc':
            return self._load_doc(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return self._load_excel(file_path)
        elif suffix in ['.pptx', '.ppt']:
            return self._load_pptx(file_path)

        # 基础加载器
        loaders = {
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.markdown': UnstructuredMarkdownLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
        }

        loader_class = loaders.get(suffix)
        if not loader_class:
            print(f"不支持的文件类型: {suffix}")
            return None

        try:
            if suffix == '.json':
                loader = loader_class(str(file_path), jq_schema='.')
            elif suffix == '.pdf':
                loader = loader_class(str(file_path))
            elif suffix == '.xlsx' or suffix == '.xls':
                loader = loader_class(str(file_path))
            elif suffix == '.csv':
                # CSV 文件尝试多种编码
                documents = self._load_csv_with_encoding(file_path)
                if documents:
                    documents = self._add_version_metadata(documents, file_path)
                    return documents
                return None
            else:
                loader = loader_class(str(file_path), encoding='utf-8')

            documents = loader.load()
            print(f"成功加载: {file_path.name}, 共 {len(documents)} 个文档")
            # 添加版本信息到metadata
            documents = self._add_version_metadata(documents, file_path)
            return documents
        except Exception as e:
            print(f"加载文件失败 {file_path.name}: {e}")
            return None

    def _load_csv_with_encoding(self, file_path: Path) -> List[Document]:
        """尝试多种编码加载 CSV 文件"""
        from langchain_community.document_loaders import CSVLoader

        # 尝试的编码顺序
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig', 'latin1']

        for encoding in encodings:
            try:
                loader = CSVLoader(str(file_path), encoding=encoding)
                documents = loader.load()
                print(f"成功加载: {file_path.name}, 共 {len(documents)} 个文档 (编码: {encoding})")
                return documents
            except Exception as e:
                continue

        print(f"CSV文件所有编码尝试失败: {file_path.name}")
        return None

    def _load_docx(self, file_path: Path) -> List[Document]:
        """加载 Word .docx 文件"""
        try:
            from docx import Document as DocxDocument

            docx = DocxDocument(str(file_path))
            paragraphs = []

            # 提取所有段落
            for para in docx.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)

            # 提取表格内容
            for table in docx.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text)

            if paragraphs:
                content = '\n\n'.join(paragraphs)
                doc = Document(
                    page_content=content,
                    metadata={'source': str(file_path), 'type': 'docx'}
                )
                print(f"成功加载: {file_path.name}")
                # 添加版本信息
                return self._add_version_metadata([doc], file_path)
            else:
                print(f"Word文件内容为空: {file_path.name}")
                return None
        except Exception as e:
            print(f"加载Word文件失败 {file_path.name}: {e}")
            return None

    def _load_doc(self, file_path: Path) -> List[Document]:
        """加载 Word .doc 文件（旧格式）"""
        try:
            # 尝试使用 pywin32 读取 .doc 文件
            import win32com.client
            import os

            # 创建 Word 应用实例
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False

            try:
                # 打开文档
                doc = word.Documents.Open(str(file_path.absolute()))
                paragraphs = []

                # 提取所有段落
                for para in doc.Paragraphs:
                    text = para.Range.Text.strip()
                    if text:
                        paragraphs.append(text)

                # 释放文档
                doc.Close(False)

                if paragraphs:
                    content = '\n\n'.join(paragraphs)
                    doc = Document(
                        page_content=content,
                        metadata={'source': str(file_path), 'type': 'doc'}
                    )
                    print(f"成功加载: {file_path.name}")
                    return self._add_version_metadata([doc], file_path)
                else:
                    print(f"Word文件内容为空: {file_path.name}")
                    return None
            finally:
                word.Quit()
        except ImportError:
            print(f"读取 .doc 文件需要 pywin32 库，请运行: pip install pywin32")
            print(f"或者将 {file_path.name} 转换为 .docx 格式")
            return None
        except Exception as e:
            print(f"加载Word .doc文件失败 {file_path.name}: {e}")
            return None

    def _load_excel(self, file_path: Path) -> List[Document]:
        """加载 Excel 文件"""
        try:
            import pandas as pd

            content_parts = []

            # 读取所有sheet
            excel_file = pd.ExcelFile(str(file_path))
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=500)
                content_parts.append(f"\n=== Sheet: {sheet_name} ===\n")
                content_parts.append(df.to_string())

            if content_parts:
                content = '\n'.join(content_parts)
                doc = Document(
                    page_content=content,
                    metadata={'source': str(file_path), 'type': 'excel'}
                )
                print(f"成功加载: {file_path.name}")
                # 添加版本信息
                return self._add_version_metadata([doc], file_path)
            else:
                print(f"Excel文件内容为空: {file_path.name}")
                return None
        except Exception as e:
            print(f"加载Excel文件失败 {file_path.name}: {e}")
            return None

    def _load_pptx(self, file_path: Path) -> List[Document]:
        """加载 PowerPoint .pptx 文件"""
        try:
            from pptx import Presentation

            prs = Presentation(str(file_path))
            paragraphs = []

            # 提取所有幻灯片的文本
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = [f"\n--- Slide {slide_num} ---\n"]

                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            if para.text.strip():
                                slide_text.append(para.text)

                    # 处理表格
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = []
                            for cell in row.cells:
                                if cell.text.strip():
                                    row_text.append(cell.text)
                            if row_text:
                                slide_text.append(' | '.join(row_text))

                if len(slide_text) > 1:  # 除了标题还有内容
                    paragraphs.append(''.join(slide_text))

            if paragraphs:
                content = '\n\n'.join(paragraphs)
                doc = Document(
                    page_content=content,
                    metadata={'source': str(file_path), 'type': 'pptx'}
                )
                print(f"成功加载: {file_path.name}")
                # 添加版本信息
                return self._add_version_metadata([doc], file_path)
            else:
                print(f"PPT文件内容为空: {file_path.name}")
                return None
        except Exception as e:
            print(f"加载PPT文件失败 {file_path.name}: {e}")
            return None

    def _load_xmind(self, file_path: Path) -> List[Document]:
        """加载 XMind 思维导图文件"""
        try:
            from xmindparser import xmind_to_dict

            # xmind_to_dict 返回结构: {'topics': [...], 'title': '...'}
            data = xmind_to_dict(str(file_path))

            # 转换为文本
            text = self._parse_xmind_dict(data)

            if text:
                doc = Document(
                    page_content=text,
                    metadata={'source': str(file_path), 'type': 'xmind'}
                )
                print(f"成功加载: {file_path.name}")
                # 添加版本信息
                return self._add_version_metadata([doc], file_path)
            else:
                print(f"XMind文件内容为空: {file_path.name}")
                return None
        except Exception as e:
            print(f"加载XMind文件失败 {file_path.name}: {e}")
            return None

    def _parse_xmind_dict(self, data, level=0) -> str:
        """递归解析 XMind 数据为文本"""
        lines = []

        if isinstance(data, dict):
            # 获取标题
            title = data.get('title', '')
            if title:
                prefix = '#' * min(level + 1, 6)
                lines.append(f"{prefix} {title}")

            # 处理子主题
            topics = data.get('topics', [])
            if isinstance(topics, dict):
                topics = [topics]
            for topic in topics:
                if isinstance(topic, dict):
                    lines.append(self._parse_xmind_dict(topic, level + 1))

        elif isinstance(data, list):
            for item in data:
                lines.append(self._parse_xmind_dict(item, level))

        return '\n'.join([l for l in lines if l])

    def _load_vsdx(self, file_path: Path) -> List[Document]:
        """加载 Visio vsdx 文件"""
        try:
            content = []
            with zipfile.ZipFile(file_path, 'r') as zf:
                # vsdx 文件中的内容在 document.xml 中
                if 'document.xml' in zf.namelist():
                    with zf.open('document.xml') as f:
                        xml_content = f.read().decode('utf-8')
                        text = self._extract_text_from_xml(xml_content)
                        if text:
                            content.append(text)

                # 页面信息
                if 'documentRelationship.xml' in zf.namelist():
                    pass  # 可以扩展处理页面信息

            if content:
                text = '\n\n'.join(content)
                doc = Document(
                    page_content=text,
                    metadata={'source': str(file_path), 'type': 'vsdx'}
                )
                print(f"成功加载: {file_path.name}")
                # 添加版本信息
                return self._add_version_metadata([doc], file_path)
            else:
                print(f"Visio文件内容为空: {file_path.name}")
                return None
        except Exception as e:
            print(f"加载Visio文件失败 {file_path.name}: {e}")
            return None

    def _extract_text_from_xml(self, xml_content: str) -> str:
        """从 XML 中提取文本"""
        import re
        # 提取 <a:t> 标签中的文本（Office Open XML 格式）
        texts = re.findall(r'<a:t[^>]*>([^<]*)</a:t>', xml_content, re.IGNORECASE)
        return '\n'.join(texts)

    def load_directory(self) -> List:
        """加载目录下的所有文档"""
        all_docs = []
        supported = set(get_supported_extensions())

        # 获取所有文件
        all_files = list(self.docs_dir.rglob('*'))
        total = len([f for f in all_files if f.is_file()])
        processed = 0

        print(f"开始加载文档，共 {total} 个文件...")

        for file_path in all_files:
            if not file_path.is_file():
                continue

            # 跳过不支持的文件类型
            if file_path.suffix.lower() not in supported:
                continue

            processed += 1
            print(f"[{processed}/{total}] 加载: {file_path.name}")

            try:
                docs = self.load_file(file_path)
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                print(f"加载失败 {file_path.name}: {e}")

        print(f"文档加载完成，共 {len(all_docs)} 个文档对象")
        return all_docs


def get_supported_extensions():
    """获取支持的文件扩展名"""
    return ['.txt', '.md', '.markdown', '.pdf', '.csv', '.json', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.xmind', '.vsdx']
