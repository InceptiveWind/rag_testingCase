"""
LLM提供商 - 使用Ollama本地模型
"""

from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


class OllamaProvider:
    """Ollama LLM提供商"""

    def __init__(
        self,
        model: str = "qwen3.5:9b-q8_0",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.chat_model = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
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
- 测试数据"""

        return self.chat(user_prompt, system_prompt=system_prompt)

    def check_connection(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            self.chat_model.invoke([HumanMessage(content="hello")])
            return True
        except Exception as e:
            print(f"Ollama连接失败: {e}")
            return False

    def chat_with_history(self, query: str, history: List[Dict[str, str]]) -> str:
        """多轮对话

        Args:
            query: 当前问题
            history: 对话历史，格式为 [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            模型回复
        """
        messages = []

        # 添加历史消息
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))

        # 添加当前问题
        messages.append(HumanMessage(content=query))

        response = self.chat_model.invoke(messages)
        return response.content
