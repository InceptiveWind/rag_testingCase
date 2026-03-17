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
EMBEDDING_MODEL = "D:/models/m3e-base"  # 本地embedding模型路径

# LLM提供商配置
LLM_PROVIDER = "volcano"  # LLM提供商: "ollama" | "volcano"

# Ollama配置（当 LLM_PROVIDER = "ollama" 时生效）
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:9b-q8_0"  # 或其他你下载的模型

# 火山引擎配置（当 LLM_PROVIDER = "volcano" 时生效）
# 火山引擎API Key（从火山引擎控制台获取）
VOLCANO_API_KEY = "87c49200-b90a-44bf-80e9-570969401a81"
# 火山引擎API地址
VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# 火山引擎模型
#VOLCANO_MODEL = "doubao-seed-2-0-code"
VOLCANO_MODEL = "doubao-seed-2-0-pro"

# 测试用例生成配置
CHUNK_SIZE = 1000  # 文档分块大小
CHUNK_OVERLAP = 200  # 分块重叠大小
TOP_K = 5  # 检索Top-K个相关文档
MAX_TOKENS = 4096  # LLM最大输出token数

# 检索配置
USE_HYBRID_RETRIEVAL = True  # 使用混合检索 (BM25 + 向量)
USE_RERANK = True  # 使用重排序
VECTOR_WEIGHT = 0.5  # 向量检索权重
BM25_WEIGHT = 0.3  # BM25权重
RERANK_TOP_K = 10  # 重排序候选数量

# 预处理配置
ENABLE_PREPROCESSOR = False  # 临时禁用文档预处理
ENABLE_LLM_TAG = False  # 默认关闭LLM标签，耗时较长
