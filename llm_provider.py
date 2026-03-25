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

    def check_connection(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            self.chat_model.invoke([HumanMessage(content="hello")])
            return True
        except Exception as e:
            print(f"Ollama连接失败: {e}")
            return False

