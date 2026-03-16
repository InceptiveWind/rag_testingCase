"""
文档加载器 - 支持多种格式的文档加载
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)


class DocumentLoader:
    """文档加载器"""

    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: Path):
        """加载单个文件"""
        suffix = file_path.suffix.lower()

        loaders = {
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.markdown': UnstructuredMarkdownLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
        }

        loader_class = loaders.get(suffix)
        if not loader_class:
            print(f"不支持的文件类型: {suffix}")
            return None

        try:
            if suffix == '.json':
                loader = loader_class(str(file_path), jq_schema='.')
            else:
                loader = loader_class(str(file_path), encoding='utf-8')

            documents = loader.load()
            print(f"成功加载: {file_path.name}, 共 {len(documents)} 个文档")
            return documents
        except Exception as e:
            print(f"加载文件失败 {file_path.name}: {e}")
            return None

    def load_directory(self) -> List:
        """加载目录下的所有文档"""
        all_docs = []

        for file_path in self.docs_dir.rglob('*'):
            if file_path.is_file():
                docs = self.load_file(file_path)
                if docs:
                    all_docs.extend(docs)

        return all_docs


def get_supported_extensions():
    """获取支持的文件扩展名"""
    return ['.txt', '.md', '.markdown', '.pdf', '.csv', '.json', '.docx', '.doc', '.xlsx', '.xls']
