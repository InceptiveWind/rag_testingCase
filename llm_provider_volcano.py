"""
LLM提供商 - 火山引擎（方舟大模型）
"""

from typing import Optional, List, Dict, Any, Union
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

    def check_connection(self) -> bool:
        """检查火山引擎服务是否可用"""
        try:
            response = self.chat_model.invoke([HumanMessage(content="hello")])
            return bool(response and response.content)
        except Exception as e:
            print(f"火山引擎连接失败: {e}")
            return False

    def chat_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: str = "auto"
    ) -> List[Dict[str, Any]]:
        """使用 Function Calling 进行对话（使用原生API）

        Args:
            message: 用户消息
            tools: 工具定义列表，每个工具包含 name, description, parameters
            system_prompt: 系统提示词
            tool_choice: 工具选择策略，"auto" 或 "required"

        Returns:
            函数调用结果列表，每个元素包含 tool_call_id, name, arguments
        """
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # 转换 tools 格式（火山引擎需要 type: "function" 外层）
        volcano_tools = []
        for tool in tools:
            if "type" not in tool:
                volcano_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                })
            else:
                volcano_tools.append(tool)

        # 构建请求体
        tool_choice_config = None
        if tool_choice == "required" and volcano_tools:
            tool_choice_config = {
                "type": "function",
                "function": {"name": volcano_tools[0]["function"]["name"]}
            }

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": volcano_tools
        }
        if tool_choice_config:
            payload["tool_choice"] = tool_choice_config

        print(f"  调用火山引擎API (Function Calling)...")

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # 解析 tool_calls
            tool_calls = result.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])

            results = []
            for tool_call in tool_calls:
                func = tool_call.get('function', {})
                results.append({
                    'tool_call_id': tool_call.get('id'),
                    'name': func.get('name'),
                    'arguments': func.get('arguments')
                })

            print(f"  API返回 {len(results)} 个工具调用")
            return results

        except requests.exceptions.Timeout:
            print("  API调用超时")
            return []
        except Exception as e:
            print(f"  API调用失败: {e}")
            return []

    def chat_streaming(
        self,
        message: str,
        system_prompt: Optional[str] = None
    ):
        """流式对话（不使用 Function Calling，直接获取文本输出）

        Args:
            message: 用户消息
            system_prompt: 系统提示词

        Yields:
            生成的文本片段
        """
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # 构建请求体（流式）
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=300)

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data = line_text[6:]
                        if data == '[DONE]':
                            break
                        # 解析 SSE 格式
                        import json as json_module
                        try:
                            chunk_data = json_module.loads(data)
                            delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                        except:
                            continue
        except Exception as e:
            print(f"  流式调用失败: {e}")
            return
