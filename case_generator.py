"""
测试用例生成器 - 根据检索到的知识生成测试用例
支持批量生成大量测试用例
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union
from langchain_core.documents import Document

# Excel支持
try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, Border, Side

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


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
            batch_size: int = 10,
            max_retries: int = 2,
            examples: str = ""
    ) -> str:
        """
        生成测试用例（支持大批量）

        Args:
            query: 查询/需求
            context_docs: 检索到的上下文文档
            num_cases: 需要生成的测试用例数量
            batch_size: 每批生成的用例数量
            max_retries: 每批最大重试次数
            examples: 用例示例，用于指导生成格式和风格

        Returns:
            生成的测试用例内容
        """
        print(f"\n开始生成测试用例（目标: {num_cases} 个，每批 {batch_size} 个）...")

        all_test_cases = []
        existing_names = []  # 记录已生成的用例名，避免重复

        # 持续生成直到达到目标数量
        batch_num = 0
        while len(all_test_cases) < num_cases:
            current_batch_size = min(batch_size, num_cases - len(all_test_cases))
            start_idx = len(all_test_cases) + 1

            print(
                f"  生成第 {batch_num + 1} 批 (目标: {current_batch_size} 个用例，已获取: {len(all_test_cases)}/{num_cases})...")

            batch_cases = []
            retry_count = 0

            # 重试直到获得足够用例或达到最大重试次数
            while len(batch_cases) < current_batch_size and retry_count < max_retries:
                # 请求数量等于目标数量
                request_size = current_batch_size - len(batch_cases)
                current_start_idx = start_idx + len(batch_cases)

                temp_cases = self._generate_batch(
                    query, context_docs,
                    start_idx=current_start_idx,
                    batch_size=request_size,
                    examples=examples,
                    existing_names=existing_names
                )

                if temp_cases:
                    # 过滤掉与已有用例重复的（根据name判断）
                    new_cases = []
                    for tc in temp_cases:
                        name = tc.get('name', '').strip()
                        if name and name not in existing_names:
                            new_cases.append(tc)
                            existing_names.append(name)
                        elif name:
                            print(f"    过滤掉重复用例: {name}")

                    batch_cases.extend(new_cases)
                    print(
                        f"    第 {retry_count + 1} 次尝试: 请求{request_size}个, 获取 {len(temp_cases)} 个, 去重后 {len(new_cases)} 个")
                else:
                    print(f"    第 {retry_count + 1} 次尝试: 获取 0 个用例")

                retry_count += 1

            if batch_cases:
                all_test_cases.extend(batch_cases)
                print(f"  本批实际获取: {len(batch_cases)} 个用例")
            else:
                print(f"  第 {batch_num + 1} 批生成失败")

            print(f"  当前累计: {len(all_test_cases)}/{num_cases} 个测试用例")
            batch_num += 1

        if not all_test_cases:
            return "错误：无法生成测试用例，请检查LLM服务是否正常运行"

        # 去除重复的用例（根据name去重，保留第一个）
        unique_cases = []
        seen_names = set()
        for tc in all_test_cases:
            name = tc.get('name', '').strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_cases.append(tc)
            elif name:
                print(f"  去除重复用例: {name}")

        print(f"\n总计生成: {len(all_test_cases)} 个测试用例，去重后: {len(unique_cases)} 个")

        return unique_cases

    def _generate_batch(
            self,
            query: str,
            context_docs: List[Document],
            start_idx: int = 1,
            batch_size: int = 10,
            examples: str = "",
            existing_names: List[str] = None
    ) -> List[Dict]:
        """生成一批测试用例"""
        if existing_names is None:
            existing_names = []

        print(f"  _generate_batch: 开始生成 {batch_size} 个用例")
        # 构建上下文
        context = "\n\n".join([
            f"文档{i + 1}:\n{doc.page_content[:1000]}"  # 限制每个文档长度
            for i, doc in enumerate(context_docs[:5])  # 最多5个文档
        ])

        # 构建示例部分
        examples_section = ""
        if examples:
            examples_section = "\n参考示例:\n" + examples + "\n"

        # 构建已存在用例的提示
        existing_section = ""
        if existing_names:
            names_str = "、".join([f'"{n}"' for n in existing_names[:20]])  # 最多显示20个
            existing_section = f"\n注意：以下用例名已经存在，请勿生成重复的：{names_str}\n"

        prompt = """基于以下需求和知识库内容，生成 {batch_size} 个测试用例。

需求: {query}
{examples_section}
知识库内容:
{context}
{existing_section}

重要格式要求（必须严格遵守）：
1. 输出必须是标准JSON数组格式
2. 每个元素是一个对象，必须包含以下6个键：name、step_id、step、precondition、priority、expected
3. name键的值只填写用例名称，不要包含任何其他内容
4. step_id键的值只填写数字（1,2,3...），表示步骤组编号
5. step键的值填写该步骤组的多个小步骤，用数字序号表示，步骤之间用"\\n"分隔（如："1. 打开登录页面\\n2. 输入用户名"）
6. precondition键的值只填写前置条件
7. priority键的值只填写"高"或"中"或"低"
8. expected键的值只填写预期结果
9. 同一个用例可以有多个步骤组，step_id=1的行填写name和priority，step_id>1的行name和priority填空
10. 不要在任何键的值中包含实际换行符（用\\n代替）
11. 不要生成与已有用例名重复的测试用例
12. 不要在返回中包含任何可导致json解析失败的符号

请按以下精确JSON格式输出，只输出数组，不要有任何其他内容：
[{{"name":"登录功能","step_id":1,"step":"1. 打开登录页面\\n2. 输入用户名\\n3. 输入密码","precondition":"系统正常运行","priority":"高","expected":"登录成功"}},{{"name":"","step_id":2,"step":"1. 点击登录按钮\\n2. 检查跳转","precondition":"","priority":"","expected":"跳转首页"}}]""".format(
            batch_size=batch_size,
            query=query,
            examples_section=examples_section,
            existing_section=existing_section,
            context=context
        )

        try:
            response = self.llm_provider.chat(prompt)

            # 打印完整原始响应用于调试
            print(f"  LLM原始响应:\n{response[:2000]}")

            # 解析JSON响应
            cases = self._parse_json_response(response, expected_count=batch_size)
            print(f"  解析得到 {len(cases)} 条用例")
            for i, c in enumerate(cases[:5]):
                print(
                    f"    {i + 1}. name='{c.get('name', '')}', step_id={c.get('step_id')}, step='{c.get('step', '')}', precondition='{c.get('precondition', '')}', priority='{c.get('priority', '')}', expected='{c.get('expected', '')}'")
            return cases
        except Exception as e:
            import traceback
            print(f"  批次生成失败: {e}")
            traceback.print_exc()
            return []

    def _parse_json_response(self, response: str, expected_count: int = 5) -> List[Dict]:
        """解析JSON响应"""
        # 清理响应，去除markdown代码块标记
        cleaned_response = response.strip()
        if cleaned_response.startswith('```'):
            # 去除 ```json 或 ``` 标记
            lines = cleaned_response.split('\n')
            cleaned_response = '\n'.join(lines[1:])
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

        # 处理LLM返回实际换行符的情况：将字符串值中的实际换行符替换为\\n
        cleaned_response = self._fix_json_newlines(cleaned_response)

        # 尝试直接解析
        try:
            cases = json.loads(cleaned_response)
            if isinstance(cases, list):
                # 修复：step字段中的真实换行符替换为\\n字符串
                for case in cases:
                    if 'step' in case and isinstance(case['step'], str):
                        case['step'] = case['step'].replace('\n', '\\n')
                print(f"  成功解析JSON，获取 {len(cases)} 个用例")
                return cases
            else:
                return [cases]
        except Exception as e:
            print(f"  JSON解析失败: {e}")
            # JSON解析失败，尝试修复截断的JSON
            cases = self._fix_truncated_json(cleaned_response, expected_count)
            if cases:
                return cases

        try:
            # 尝试提取JSON数组
            match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
            if match:
                json_str = match.group()
                cases = json.loads(json_str)
                if isinstance(cases, list):
                    print(f"  成功解析JSON，获取 {len(cases)} 个用例")
                    # 修复：step字段中的真实换行符替换为\\n字符串
                    for case in cases:
                        if 'step' in case and isinstance(case['step'], str):
                            case['step'] = case['step'].replace('\n', '\\n')
                    return cases
                else:
                    return [cases]

            # 尝试直接解析
            cases = json.loads(cleaned_response)
            if isinstance(cases, list):
                return cases
            else:
                return [cases]
        except Exception as e:
            print(f"  JSON解析失败: {e}")
            # JSON解析失败，尝试修复截断的JSON
            return self._fix_truncated_json(cleaned_response, expected_count)

    def _fix_json_newlines(self, json_str: str) -> str:
        """修复JSON中未转义的换行符"""
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(json_str):
            char = json_str[i]
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"':
                result.append(char)
                in_string = not in_string
            elif char == '\n' and in_string:
                # 在字符串内的换行符需要转义
                result.append('\\n')
            elif char == '\r':
                # 忽略回车符
                pass
            else:
                result.append(char)
            i += 1
        return ''.join(result)

    def _fix_truncated_json(self, json_str: str, expected_count: int) -> List[Dict]:
        """尝试修复截断的JSON"""
        cases = []

        # 方法1：找到最后一个完整的对象
        try:
            array_start = json_str.find('[')
            if array_start >= 0:
                # 找到最后一个 }
                last_brace = json_str.rfind('}')
                if last_brace > array_start:
                    fixed_json = json_str[array_start:last_brace + 1] + ']'
                    # 修复未关闭的字符串
                    fixed_json = self._close_unclosed_strings(fixed_json)
                    fixed_cases = json.loads(fixed_json)
                    if isinstance(fixed_cases, list) and len(fixed_cases) > 0:
                        print(f"  修复截断JSON成功，获取 {len(fixed_cases)} 个用例")
                        return fixed_cases
        except Exception as e:
            print(f"  修复尝试1失败: {e}")

        # 方法2：正则提取每个完整对象
        try:
            case_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL)
            for block in case_blocks[:expected_count]:
                case = self._extract_case_from_block(block)
                if case and case.get('name'):
                    cases.append(case)
            if cases:
                print(f"  正则提取获取 {len(cases)} 个用例")
                return cases
        except Exception as e:
            print(f"  修复尝试2失败: {e}")

        return cases

    def _close_unclosed_strings(self, json_str: str) -> str:
        """关闭JSON中未关闭的字符串"""
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(json_str):
            char = json_str[i]
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"':
                result.append(char)
                in_string = not in_string
            elif not in_string:
                result.append(char)
            i += 1

        # 如果字符串未关闭，添加闭合引号和括号
        if in_string:
            result.append('"')
        return ''.join(result)

    def _extract_case_from_block(self, block: str) -> Dict:
        """从JSON块中提取用例信息"""
        case = {}
        try:
            # 尝试直接解析这个块
            obj = json.loads(block)
            return obj
        except:
            # 回退到正则提取
            name_match = re.search(r'["\']?name["\']?\s*:\s*["\']([^"\']*)["\']', block)
            if name_match:
                case['name'] = name_match.group(1).strip()
            step_id_match = re.search(r'["\']?step_id["\']?\s*:\s*(\d+)', block)
            if step_id_match:
                case['step_id'] = int(step_id_match.group(1))
            step_match = re.search(r'["\']?step["\']?\s*:\s*["\']([^"\']*)["\']', block, re.DOTALL)
            if step_match:
                case['step'] = step_match.group(1).replace('\n', '\\n')
            precondition_match = re.search(r'["\']?precondition["\']?\s*:\s*["\']([^"\']*)["\']', block)
            if precondition_match:
                case['precondition'] = precondition_match.group(1).strip()
            priority_match = re.search(r'["\']?priority["\']?\s*:\s*["\']([^"\']*)["\']', block)
            if priority_match:
                case['priority'] = priority_match.group(1).strip()
            expected_match = re.search(r'["\']?expected["\']?\s*:\s*["\']([^"\']*)["\']', block)
            if expected_match:
                case['expected'] = expected_match.group(1).strip()
        return case

    def _merge_multi_step_cases(self, cases: List[Dict]) -> List[Dict]:
        """合并同一个用例的多个step_id行"""
        # 按name分组，收集所有step_id>1的步骤
        name_to_steps = {}  # name -> [(step_id, step, expected), ...]
        name_to_main = {}    # name -> step_id=1的完整用例

        for tc in cases:
            name = tc.get('name', '')
            step_id = tc.get('step_id', 1)
            step = tc.get('step', '')
            expected = tc.get('expected', '')

            if not name:  # name为空，跳过
                continue

            if step_id == 1:
                name_to_main[name] = tc.copy()
                name_to_steps[name] = []
            else:
                if name in name_to_steps:
                    name_to_steps[name].append((step_id, step, expected))

        # 合并多步骤
        merged_cases = []
        for name, main_case in name_to_main.items():
            # 添加主用例（step_id=1）
            merged_cases.append(main_case)
            # 添加后续步骤
            if name in name_to_steps and name_to_steps[name]:
                # 按step_id排序
                name_to_steps[name].sort(key=lambda x: x[0])
                for step_id, step, expected in name_to_steps[name]:
                    merged_cases.append({
                        'name': '',  # 后续步骤name为空
                        'step_id': step_id,
                        'step': step,
                        'precondition': '',
                        'priority': '',
                        'expected': expected
                    })

        # 添加没有对应step_id=1的孤立行
        for tc in cases:
            name = tc.get('name', '')
            step_id = tc.get('step_id', 1)
            if not name and step_id and step_id > 1:
                # 检查是否已添加
                merged_cases.append(tc)

        print(f"  合并多步骤: {len(cases)} -> {len(merged_cases)} 行")
        return merged_cases

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
        new_cases = self._generate_batch(query, context_docs, start_idx=len(existing_cases) + 1, examples="")

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

    def save_to_excel(self, cases: Union[List[Dict], str], filename: str = None) -> Path:
        """保存测试用例到Excel文件

        Args:
            cases: 测试用例列表（Dict）或Markdown内容（str）
            filename: 文件名

        Returns:
            保存的文件路径
        """
        if not EXCEL_AVAILABLE:
            print("警告: openpyxl未安装，保存为Markdown格式")
            if isinstance(cases, str):
                return self.save_to_file(cases)
            return self.save_to_file(self._format_cases(cases))

        # 如果传入的是字符串（Markdown格式），先解析为用例列表
        if isinstance(cases, str):
            cases = self._parse_markdown_cases(cases)

        # 过滤无效用例：必须要有step字段
        valid_cases = [c for c in cases if c.get('step') and c.get('step').strip()]
        if len(valid_cases) < len(cases):
            print(f"  过滤掉 {len(cases) - len(valid_cases)} 个无效用例")
        cases = valid_cases

        # 合并多步骤用例：同一个name的step_id>1的行合并到step_id=1的行
        cases = self._merge_multi_step_cases(cases)

        # 保存前统计用例数量
        step1_cases = [c for c in cases if c.get('step_id') == 1]
        unique_names = set(c.get('name', '') for c in cases if c.get('name', '').strip())
        print(
            f"  保存到Excel前: 共 {len(cases)} 行, step_id=1用例: {len(step1_cases)} 个, 唯一name: {len(unique_names)} 个")

        if not cases:
            print("警告: 没有测试用例可保存")
            return self.save_to_file("无测试用例")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_case_{timestamp}.xlsx"

        filepath = self.output_dir / filename

        # 读取模板文件
        template_path = Path(__file__).parent / "templates" / "测试用例模板.xlsx"
        try:
            wb = openpyxl.load_workbook(str(template_path))
            ws = wb.active
            # 删除模板原有的空数据行，只保留表头
        except Exception as e:
            print(f"警告: 无法读取模板文件 {template_path}: {e}")
            # 创建新工作簿
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "测试用例"

        # 找到数据开始行（跳过表头）
        start_row = 2

        # 计算每个用例的编号（按name计数，每个唯一的name算一个用例）
        # 同一个name的多个step_id行共享同一个编号
        case_number = 0
        seen_names = set()  # 用于记录已经编号过的name
        for tc in cases:
            name = tc.get('name', '')
            step_id = tc.get('step_id')

            # name为空时不跳过（step_id>1的行需要正常写入，只是name留空）

            # 如果这个name还没编号过，则编号
            if name not in seen_names:
                case_number += 1
                seen_names.add(name)
                tc['_case_number'] = case_number
            else:
                # 已编号的name，不再编号
                tc['_case_number'] = None

        # 写入测试用例数据
        for idx, tc in enumerate(cases):
            row_idx = start_row + idx

            # 获取字段值
            step_id = tc.get('step_id')  # 默认None，不是1
            name = tc.get('name', '')
            precondition = tc.get('precondition', '')
            priority = tc.get('priority', '')
            step = tc.get('step', '')
            expected = tc.get('expected', '')
            case_num = tc.get('_case_number')

            # 编号（第1列）- 只在 step_id=1 时填写（表示用例序号）
            if case_num is not None:
                ws.cell(row=row_idx, column=1, value=case_num)
            else:
                ws.cell(row=row_idx, column=1, value='')

            # 标题（第4列）- 只在 step_id=1 时填写
            if step_id == 1:
                ws.cell(row=row_idx, column=4, value=name)
            else:
                ws.cell(row=row_idx, column=4, value='')

            # 前置条件（第6列）- 每一行填写
            ws.cell(row=row_idx, column=6, value=precondition)

            # 优先级（第7列）- 只在 step_id=1 时填写
            if step_id == 1:
                ws.cell(row=row_idx, column=7, value=priority)
            else:
                ws.cell(row=row_idx, column=7, value='')

            # 步骤ID（第8列）- 只有 step_id 存在时才填写
            if step_id is not None:
                ws.cell(row=row_idx, column=8, value=step_id)
            else:
                ws.cell(row=row_idx, column=8, value='')

            # 步骤（第9列）- 用换行符分隔多个步骤
            ws.cell(row=row_idx, column=9, value=step)

            # 期望结果（第10列）
            ws.cell(row=row_idx, column=10, value=expected)

            # 设置行高以适应内容（根据换行符数量调整）
            max_lines = max(step.count('\n'), expected.count('\n')) + 1
            ws.row_dimensions[row_idx].height = max(20, max_lines * 20)

        # 删除最后一条用例之后的空行（保留表头）
        last_case_row = len(cases) + 1
        while ws.max_row > last_case_row:
            ws.delete_rows(ws.max_row)

        # 保存文件
        wb.save(filepath)
        print(f"\n测试用例已保存到: {filepath}")
        return filepath

    def _parse_markdown_cases(self, markdown_text: str) -> List[Dict]:
        """解析Markdown格式的测试用例为字典列表"""
        cases = []

        # 提取每个测试用例块（从 ## 测试用例 到下一个 ## 或结尾）
        pattern = r'##\s*测试用例\s*\d+:\s*(.+?)(?=\n##\s*测试用例|\Z)'
        matches = re.findall(pattern, markdown_text, re.DOTALL)

        for match in matches:
            case = {'name': '', 'steps': [], 'expected': '', 'priority': ''}

            # 按行处理
            lines = match.strip().split('\n')
            if not lines:
                continue

            # 第一行是标题
            case['name'] = lines[0].strip()

            # 后续行提取steps和expected
            current_field = None
            for line in lines[1:]:
                line_stripped = line.strip()

                # 跳过空行和分隔符
                if not line_stripped or line_stripped == '---':
                    continue

                # 检测字段标记（如 "**测试步骤**:" 或 "**预期结果**: ..."）
                # 需要检测是否同时包含字段名和内容（用冒号分隔）
                if '测试步骤' in line_stripped:
                    current_field = 'steps'
                    # 如果冒号后面还有内容，添加到steps
                    if ':' in line_stripped:
                        parts = line_stripped.split(':', 1)
                        if len(parts) > 1 and parts[1].strip():
                            step = parts[1].strip().replace('**', '').strip()
                            if step:
                                case['steps'].append(step)
                    continue
                elif '预期结果' in line_stripped:
                    current_field = 'expected'
                    # 如果冒号后面还有内容，添加到expected
                    if ':' in line_stripped:
                        parts = line_stripped.split(':', 1)
                        if len(parts) > 1 and parts[1].strip():
                            exp = parts[1].strip().replace('**', '').strip()
                            if exp:
                                case['expected'] = exp
                    continue

                # 提取内容
                if current_field == 'steps':
                    # 去掉序号如 "1. " 或 "1. "
                    step = re.sub(r'^\d+[.、]\s*', '', line_stripped)
                    # 去掉markdown加粗标记
                    step = step.replace('**', '').strip()
                    if step:
                        case['steps'].append(step)

                elif current_field == 'expected':
                    # 去掉markdown加粗标记
                    exp = line_stripped.replace('**', '').strip()
                    if exp:
                        if case['expected']:
                            case['expected'] += '\n' + exp
                        else:
                            case['expected'] = exp

            cases.append(case)

        return cases

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
        formatted_steps = "\n".join([f"    {i + 1}. {step}" for i, step in enumerate(steps)])

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
