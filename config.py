"""
配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 知识库配置
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "docs"  # 文档目录
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"  # 向量存储目录
CASES_OUTPUT_DIR = PROJECT_ROOT / "cases"  # 测试用例输出目录

# Chroma配置
COLLECTION_NAME = "test_knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文embedding模型

# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:9b-q8_0"  # 或其他你下载的模型

# 测试用例生成配置
CHUNK_SIZE = 1000  # 文档分块大小
CHUNK_OVERLAP = 200  # 分块重叠大小
TOP_K = 3  # 检索Top-K个相关文档
MAX_TOKENS = 4096  # LLM最大输出token数
