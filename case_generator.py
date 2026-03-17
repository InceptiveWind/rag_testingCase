"""
测试用例生成器 - 根据检索到的知识生成测试用例
支持批量生成大量测试用例
"""

import json
import re
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

    def generate(
        self,
        query: str,
        context_docs: List[Document],
        num_cases: int = 10,
        batch_size: int = 5,
        max_retries: int = 3
    ) -> str:
        """
        生成测试用例（支持大批量）

        Args:
            query: 查询/需求
            context_docs: 检索到的上下文文档
            num_cases: 需要生成的测试用例数量
            batch_size: 每批生成的用例数量
            max_retries: 每批最大重试次数

        Returns:
            生成的测试用例内容
        """
        print(f"\n开始生成测试用例（目标: {num_cases} 个，每批 {batch_size} 个）...")

        all_test_cases = []
        batches = (num_cases + batch_size - 1) // batch_size  # 计算批次数

        for batch_num in range(batches):
            current_batch_size = min(batch_size, num_cases - len(all_test_cases))
            start_idx = len(all_test_cases) + 1

            print(f"  生成第 {batch_num + 1}/{batches} 批 (目标: {current_batch_size} 个用例)...")

            batch_cases = []
            retry_count = 0

            # 重试直到获得足够用例或达到最大重试次数
            while len(batch_cases) < current_batch_size and retry_count < max_retries:
                # 调整请求数量：请求比目标多几个，以防LLM返回不足
                request_size = current_batch_size - len(batch_cases) + 2
                current_start_idx = start_idx + len(batch_cases)

                temp_cases = self._generate_batch(
                    query, context_docs,
                    start_idx=current_start_idx,
                    batch_size=request_size
                )

                if temp_cases:
                    batch_cases.extend(temp_cases)
                    print(f"    第 {retry_count + 1} 次尝试: 请求{request_size}个, 获取 {len(temp_cases)} 个用例")
                else:
                    print(f"    第 {retry_count + 1} 次尝试: 获取 0 个用例")

                retry_count += 1

            # 截取目标数量的用例
            batch_cases = batch_cases[:current_batch_size]

            if batch_cases:
                all_test_cases.extend(batch_cases)
                print(f"  本批实际获取: {len(batch_cases)} 个用例")
            else:
                print(f"  第 {batch_num + 1} 批生成失败")

            print(f"  当前累计: {len(all_test_cases)}/{num_cases} 个测试用例")

            # 如果已经达到目标，可以提前结束
            if len(all_test_cases) >= num_cases:
                break

        if not all_test_cases:
            return "错误：无法生成测试用例，请检查LLM服务是否正常运行"

        # 合并所有测试用例
        final_content = self._format_cases(all_test_cases)
        print(f"\n总计生成: {len(all_test_cases)} 个测试用例")
        return final_content

    def _generate_batch(
        self,
        query: str,
        context_docs: List[Document],
        start_idx: int = 1,
        batch_size: int = 5
    ) -> List[Dict]:
        """生成一批测试用例"""
        print(f"  _generate_batch: 开始生成 {batch_size} 个用例")
        # 构建上下文
        context = "\n\n".join([
            f"文档{i+1}:\n{doc.page_content[:1000]}"  # 限制每个文档长度
            for i, doc in enumerate(context_docs[:5])  # 最多5个文档
        ])

        prompt = f"""基于以下需求和知识库内容，生成 {batch_size} 个测试用例。

需求: {query}

知识库内容:
{context}

请为每个测试用例包含以下信息：
1. 测试用例名称
2. 前置条件
3. 测试步骤
4. 预期结果
5. 测试数据

请按以下格式输出（JSON数组格式）:
[
  {{
    "name": "测试用例名称",
    "precondition": "前置条件",
    "steps": ["步骤1", "步骤2", "步骤3"],
    "expected": "预期结果",
    "test_data": "测试数据"
  }}
]

只返回JSON数组，不要其他内容。"""

        try:
            response = self.llm_provider.chat(prompt)

            # 解析JSON响应
            cases = self._parse_json_response(response, expected_count=batch_size)
            return cases
        except Exception as e:
            print(f"  批次生成失败: {e}")
            return []

    def _parse_json_response(self, response: str, expected_count: int = 5) -> List[Dict]:
        """解析JSON响应"""
        try:
            # 尝试提取JSON数组
            # 查找 [ 和 ] 之间的内容
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group()
                cases = json.loads(json_str)
                if isinstance(cases, list):
                    print(f"  成功解析JSON，获取 {len(cases)} 个用例")
                    return cases
                else:
                    return [cases]

            # 尝试直接解析
            cases = json.loads(response)
            if isinstance(cases, list):
                return cases
            else:
                return [cases]
        except Exception as e:
            print(f"  JSON解析失败: {e}")
            # JSON解析失败，尝试正则提取
            cases = []
            # 匹配每个测试用例块 - 尝试匹配完整对象
            pattern = r'\{\s*["\']?name["\']?\s*:\s*["\']([^"\']+)["\']'
            names = re.findall(pattern, response)

            print(f"  正则提取到 {len(names)} 个用例名称")

            # 使用实际匹配到的数量，而不是固定5个
            for name in names[:min(expected_count, len(names))]:
                cases.append({
                    "name": name.strip(),
                    "precondition": "无",
                    "steps": ["步骤1", "步骤2"],
                    "expected": "符合预期",
                    "test_data": "无"
                })

            # 如果仍然没有提取到，尝试更宽松的匹配
            if not cases:
                # 尝试匹配任何看起来像测试用例的内容
                pattern2 = r'(?:测试用例|testcase|case)[：:\s]*([^,\n]{2,30})'
                names2 = re.findall(pattern2, response, re.IGNORECASE)
                for name in names2[:min(expected_count, len(names2))]:
                    cases.append({
                        "name": name.strip(),
                        "precondition": "无",
                        "steps": ["步骤1", "步骤2"],
                        "expected": "符合预期",
                        "test_data": "无"
                    })

            return cases

    def _format_cases(self, cases: List[Dict]) -> str:
        """格式化测试用例为Markdown"""
        md_content = f"# 集成测试用例\n\n"
        md_content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += f"共 {len(cases)} 个测试用例\n\n"
        md_content += "---\n\n"

        for i, tc in enumerate(cases, 1):
            md_content += f"## 测试用例 {i}: {tc.get('name', '未命名')}\n\n"

            precond = tc.get('precondition', '无')
            if precond and precond != '无':
                md_content += f"**前置条件**: {precond}\n\n"

            md_content += "**测试步骤**:\n"
            steps = tc.get('steps', [])
            if isinstance(steps, list):
                for j, step in enumerate(steps, 1):
                    md_content += f"{j}. {step}\n"
            else:
                md_content += f"1. {steps}\n"

            md_content += f"\n**预期结果**: {tc.get('expected', '')}\n\n"

            test_data = tc.get('test_data', '')
            if test_data and test_data != '无':
                md_content += f"**测试数据**: {test_data}\n\n"

            md_content += "---\n\n"

        return md_content

    def generate_continue(
        self,
        query: str,
        context_docs: List[Document],
        existing_cases: List[Dict],
        continue_prompt: str = ""
    ) -> str:
        """
        继续生成更多测试用例（用于上下文续写）

        Args:
            query: 查询/需求
            context_docs: 上下文文档
            existing_cases: 已生成的用例
            continue_prompt: 续写提示

        Returns:
            追加新用例后的完整内容
        """
        new_cases = self._generate_batch(query, context_docs, start_idx=len(existing_cases) + 1)

        if new_cases:
            all_cases = existing_cases + new_cases
            return self._format_cases(all_cases)

        return self._format_cases(existing_cases)

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
