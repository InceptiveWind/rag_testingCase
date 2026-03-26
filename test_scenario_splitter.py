"""
测试场景化拆分器 - 将文档按测试场景维度拆分，提取测试相关实体和标签
"""

import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document


class TestScenarioSplitter:
    """测试场景化拆分器

    基于文档结构自动识别章节，提取测试场景，生成测试专用标签。
    """

    # 章节编号匹配模式：如 4.1.0, 4.1.1, 4.2.3
    SECTION_PATTERN = re.compile(r'^(\d+\.\d+\.\d+)\s*([^\d\n][^\n]*)$', re.MULTILINE)

    # 测试场景关键词
    SCENARIO_KEYWORDS = {
        'positive': ['成功', '通过', '正常', '正确', '有效', '完成', '允许'],
        'negative': ['失败', '不通过', '异常', '错误', '拒绝', '无效', '禁止', '不能', '无法', '不存在', '不等于'],
        'boundary': ['最大', '最小', '最多', '最少', '边界', '临界', '上限', '下限', '等于'],
    }

    # 触发条件关键词
    CONDITION_KEYWORDS = ['如果', '当', '当且', '假设', '假如', '一旦', '只要', '除非']

    # 需要过滤的章节/内容
    FILTER_PATTERNS = [
        r'^基本信息',
        r'^背景描述',
        r'^开发目的',
        r'^程序性能',
        r'^运行环境',
        r'^附件/样本',
        r'^审批内容',
        r'^修改历史',
        r'^版本更新',
        r'^程序测试',
    ]

    # 文档元数据表格行（基本信息的键值对）
    METADATA_TABLE_PATTERNS = [
        r'^文档编号',
        r'^项目名称',
        r'^需求名称',
        r'^程序名',
        r'^事务代码',
        r'^关键用户',
        r'^业务顾问',
        r'^需求日期',
        r'^版本',
        r'^分类',
        r'^需求类型',
        r'^开发类型',
        r'^运行频率',
        r'^优先级别',
    ]

    def __init__(self, enable_llm: bool = False, llm_provider=None):
        """
        初始化测试场景拆分器

        Args:
            enable_llm: 是否启用LLM辅助提取（暂未实现）
            llm_provider: LLM提供商
        """
        self.enable_llm = enable_llm
        self.llm_provider = llm_provider

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行测试场景化拆分

        Args:
            documents: 原始文档列表

        Returns:
            拆分后的文档列表
        """
        if not documents:
            return documents

        print(f"开始测试场景化拆分，共 {len(documents)} 个文档...")

        result = []
        for doc in documents:
            split_docs = self._split_single_document(doc)
            result.extend(split_docs)

        print(f"  场景化拆分完成: {len(documents)} 个文档 -> {len(result)} 个场景切片")
        return result

    def _split_single_document(self, doc: Document) -> List[Document]:
        """拆分单个文档"""
        content = doc.page_content
        metadata = doc.metadata.copy()

        # 1. 提取文档标题作为模块名
        module_name = self._extract_module_name(metadata.get('source', ''))

        # 2. 识别章节结构
        sections = self._extract_sections(content)

        if not sections:
            # 如果没有识别到章节，返回原始文档（带基础标签）
            metadata['module'] = module_name
            metadata['scenario_type'] = 'unknown'
            metadata['test_relevance'] = {'rules': [], 'entities': []}
            return [doc]

        # 3. 构建每个章节的文档
        result = []
        for section_info in sections:
            section_content = section_info['content']
            section_title = section_info['title']
            section_num = section_info['number']

            # 跳过过滤内容
            if self._should_filter(section_title, section_content):
                continue

            # 提取测试场景信息
            scenario_info = self._extract_scenario_info(section_content)

            # 构建切片文档
            new_metadata = metadata.copy()
            new_metadata['module'] = module_name
            new_metadata['section_number'] = section_num
            new_metadata['section_title'] = section_title
            new_metadata['scenario_type'] = scenario_info['scenario_type']
            new_metadata['test_relevance'] = scenario_info
            new_metadata['is_test_scenario'] = True

            new_doc = Document(
                page_content=section_content.strip(),
                metadata=new_metadata
            )
            result.append(new_doc)

        return result if result else [doc]

    def _extract_module_name(self, source_path: str) -> str:
        """从文件路径提取模块名"""
        if not source_path:
            return '未知模块'

        # 从路径中提取模块名
        # 例如: docs/商机模块/商机管理优化V1.6.docx -> 商机模块
        path_parts = source_path.replace('\\', '/').split('/')
        if len(path_parts) >= 2:
            return path_parts[-2] if path_parts[-2] != 'docs' else path_parts[-1].split('.')[0]

        return source_path.split('.')[0]

    def _extract_sections(self, content: str) -> List[Dict]:
        """提取文档中的章节结构"""
        sections = []

        # 匹配章节编号和标题
        # 例如: "4.1.0商机审核优化" -> number="4.1.0", title="商机审核优化"
        matches = list(self.SECTION_PATTERN.finditer(content))

        if not matches:
            return []

        for i, match in enumerate(matches):
            section_num = match.group(1)
            section_title = match.group(2).strip()

            # 获取章节内容（直到下一个章节之前）
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start_pos:end_pos]

            sections.append({
                'number': section_num,
                'title': section_title,
                'content': section_content
            })

        return sections

    def _should_filter(self, title: str, content: str) -> bool:
        """判断章节是否应该过滤"""
        # 检查标题
        for pattern in self.FILTER_PATTERNS:
            if re.search(pattern, title):
                return True

        # 检查内容是否过短（可能是标题行而非正文）
        if len(content.strip()) < 50:
            return True

        # 检查是否是元数据表格行
        lines = content.split('\n')[:5]
        metadata_count = 0
        for line in lines:
            for pattern in self.METADATA_TABLE_PATTERNS:
                if re.search(pattern, line.strip()):
                    metadata_count += 1
                    break

        # 如果前5行中超过3行是元数据，认为是基本信息区域
        if metadata_count >= 3:
            return True

        return False

    def _extract_scenario_info(self, content: str) -> Dict:
        """
        从章节内容中提取测试场景信息

        Returns:
            包含 scenario_type, rules, entities 等信息的字典
        """
        # 1. 识别场景类型
        scenario_type = self._detect_scenario_type(content)

        # 2. 提取业务规则（条件语句）
        rules = self._extract_rules(content)

        # 3. 提取测试相关实体（数据库字段）
        entities = self._extract_entities(content)

        # 4. 提取测试点
        test_points = self._extract_test_points(content)

        return {
            'scenario_type': scenario_type,
            'rules': rules,
            'entities': entities,
            'test_points': test_points
        }

    def _detect_scenario_type(self, content: str) -> str:
        """检测场景类型"""
        content_lower = content

        # 检查是否包含正向/负向/边界关键词
        positive_count = sum(1 for kw in self.SCENARIO_KEYWORDS['positive'] if kw in content_lower)
        negative_count = sum(1 for kw in self.SCENARIO_KEYWORDS['negative'] if kw in content_lower)
        boundary_count = sum(1 for kw in self.SCENARIO_KEYWORDS['boundary'] if kw in content_lower)

        max_count = max(positive_count, negative_count, boundary_count)

        if max_count == 0:
            return 'positive'  # 默认正向

        if max_count == positive_count:
            return 'positive'
        elif max_count == negative_count:
            return 'negative'
        else:
            return 'boundary'

    def _extract_rules(self, content: str) -> List[str]:
        """提取业务规则（条件语句）"""
        rules = []

        # 匹配条件语句：如果xxx，则xxx / 当xxx时 / 是否xxx
        condition_patterns = [
            r'[如果当若假设]([^。，,]+?)[，,]([^。，。]+?)[。\n]',
            r'[是否]([^。，,]+?)[，。](?:则)?([^。，。]+?)[。\n]',
            r'当([^。，,]+?)时[，,]([^。，。]+?)[。\n]',
        ]

        for pattern in condition_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                rule = match.group(0).strip()
                if len(rule) > 10 and len(rule) < 200:  # 过滤过短或过长的规则
                    rules.append(rule)

        # 去重
        seen = set()
        unique_rules = []
        for rule in rules:
            if rule not in seen:
                seen.add(rule)
                unique_rules.append(rule)

        return unique_rules[:10]  # 最多保留10条规则

    def _extract_entities(self, content: str) -> List[str]:
        """提取数据库字段引用"""
        # 匹配格式: 表名.字段名 或 表名.字段名 (xxx)
        entity_pattern = r'([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)'
        matches = re.findall(entity_pattern, content.lower())

        # 过滤常见非字段词
        stop_words = {'and', 'or', 'not', 'in', 'is', 'null', 'true', 'false', 'if', 'else', 'then'}
        entities = [f"{table}.{field}" for table, field in matches if field not in stop_words]

        # 去重
        return list(set(entities))[:20]  # 最多保留20个实体

    def _extract_test_points(self, content: str) -> List[str]:
        """提取测试关注点"""
        test_points = []

        # 识别"需要/必须/应当"等关键验证词
        verify_patterns = [
            r'[需要必须应当]([^。，,]+?)[。\n]',
            r'验证([^。，,]+?)[。\n]',
            r'检查([^。，,]+?)[。\n]',
            r'确保([^。，,]+?)[。\n]',
        ]

        for pattern in verify_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                point = match.group(0).strip()
                if len(point) > 5 and len(point) < 100:
                    test_points.append(point)

        # 去重
        seen = set()
        unique_points = []
        for point in test_points:
            if point not in seen:
                seen.add(point)
                unique_points.append(point)

        return unique_points[:10]  # 最多保留10个测试点
