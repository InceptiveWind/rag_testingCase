"""
测试用例生成器 - 根据检索到的知识生成测试用例
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document


class TestCaseGenerator:
    """测试用例生成器"""

    def __init__(self, llm_provider, output_dir: Path):
        self.llm_provider = llm_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """生成测试用例"""
        print("\n正在生成测试用例...")

        # 使用LLM生成测试用例（带重试机制）
        max_retries = 3
        for attempt in range(max_retries):
            test_cases = self.llm_provider.generate_with_context(query, context_docs)
            if test_cases and len(test_cases.strip()) > 0:
                return test_cases
            print(f"生成结果为空，尝试第 {attempt + 1}/{max_retries} 次...")

        # 所有重试都失败，返回错误信息
        return "错误：无法生成测试用例，请检查LLM服务是否正常运行"

    def save_to_file(self, content: str, filename: str = None) -> Path:
        """保存测试用例到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_case_{timestamp}.md"

        filepath = self.output_dir / filename
        filepath.write_text(content, encoding='utf-8')
        print(f"\n测试用例已保存到: {filepath}")
        return filepath

    def generate_and_save(self, query: str, context_docs: List[Document]) -> Dict[str, str]:
        """生成并保存测试用例"""
        test_cases = self.generate(query, context_docs)
        filepath = self.save_to_file(test_cases)

        return {
            "content": test_cases,
            "filepath": str(filepath)
        }

    def format_as_pytest(self, test_case_content: str) -> str:
        """将测试用例格式化为pytest格式"""
        # 简单的格式化，实际使用时可以让LLM直接生成pytest格式
        pytest_template = f'''"""
Generated Integration Tests
Auto-generated based on RAG knowledge base
"""

import pytest


{test_case_content}
'''
        return pytest_template


class TestCaseFormatter:
    """测试用例格式化器"""

    @staticmethod
    def format_test_case(name: str, steps: List[str], expected: str) -> str:
        """格式化单个测试用例"""
        formatted_steps = "\n".join([f"    {i+1}. {step}" for i, step in enumerate(steps)])

        return f"""
### 测试用例: {name}

**测试步骤:**
{formatted_steps}

**预期结果:**
{expected}
"""

    @staticmethod
    def format_as_markdown(test_cases: List[Dict]) -> str:
        """格式化为Markdown格式"""
        md_content = "# 集成测试用例\n\n"
        md_content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for tc in test_cases:
            md_content += f"## {tc.get('name', '未命名测试用例')}\n\n"
            md_content += f"**前置条件**: {tc.get('precondition', '无')}\n\n"
            md_content += f"**测试步骤**:\n"
            for i, step in enumerate(tc.get('steps', []), 1):
                md_content += f"{i}. {step}\n"
            md_content += f"\n**预期结果**: {tc.get('expected', '')}\n\n"
            md_content += "---\n\n"

        return md_content
