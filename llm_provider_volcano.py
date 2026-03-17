"""
LLM提供商 - 火山引擎（方舟大模型）
"""

from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class VolcanoProvider:
    """火山引擎 LLM提供商（方舟大模型）"""

    # 火山引擎方舟API地址
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    # 支持的模型列表
    SUPPORTED_MODELS = [
        "doubao-pro-4k",        # 豆包Pro 4K
        "doubao-pro-32k",       # 豆包Pro 32K
        "doubao-lite-4k",       # 豆包Lite 4K
        "doubao-lite-32k",      # 豆包Lite 32K
        "ep-20240513",          # EP模型
    ]

    def __init__(
        self,
        model: str = "doubao-pro-4k",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        初始化火山引擎LLM提供商

        Args:
            model: 模型名称
            api_key: API密钥（火山引擎控制台获取）
            base_url: API地址（默认使用方舟API）
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化 ChatOpenAI（火山引擎兼容 OpenAI API）
        self.chat_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """单轮对话"""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=message))

        response = self.chat_model.invoke(messages)
        return response.content

    def generate(self, prompt: str) -> str:
        """生成回复（chat的别名）"""
        return self.chat(prompt)

    def generate_with_context(self, query: str, context_docs: List) -> str:
        """基于上下文生成回复"""
        # 构建上下文
        context = "\n\n".join([
            f"文档{i+1}:\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        system_prompt = """你是一个专业的测试工程师，擅长根据需求文档生成高质量的测试用例。
请根据提供的上下文信息生成测试用例，确保：
1. 测试用例覆盖关键功能点
2. 包含正常情况和异常情况
3. 使用清晰的测试用例格式
4. 测试步骤具体可执行"""

        user_prompt = f"""基于以下知识库内容，为这个功能生成集成测试用例：

查询需求: {query}

知识库内容:
{context}

请生成完整的集成测试用例，包括：
- 测试用例名称
- 前置条件
- 测试步骤
- 预期结果
"""

        return self.chat(user_prompt, system_prompt=system_prompt)

    def check_connection(self) -> bool:
        """检查火山引擎服务是否可用"""
        try:
            response = self.chat_model.invoke([HumanMessage(content="hello")])
            return bool(response and response.content)
        except Exception as e:
            print(f"火山引擎连接失败: {e}")
            return False


def create_llm_provider(provider_type: str, **kwargs):
    """
    工厂函数：创建LLM提供商

    Args:
        provider_type: 提供商类型 ("ollama" | "volcano")
        **kwargs: 其他参数

    Returns:
        LLM提供商实例
    """
    if provider_type == "volcano":
        return VolcanoProvider(**kwargs)
    elif provider_type == "ollama":
        from llm_provider import OllamaProvider
        return OllamaProvider(**kwargs)
    else:
        raise ValueError(f"不支持的LLM提供商类型: {provider_type}")
