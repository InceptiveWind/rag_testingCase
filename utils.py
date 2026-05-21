"""
公共工具函数 - 消除代码重复、提供类型注解
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def safe_json_parse(text: str, fallback: Any = None) -> Any:
    """
    安全解析 JSON 字符串
    
    Args:
        text: 待解析的文本
        fallback: 解析失败时的默认返回值
        
    Returns:
        解析后的 JSON 对象或 fallback
    """
    if not text or not text.strip():
        return fallback
        
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 尝试修复常见问题
    fixed_text = fix_json_text(text)
    
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        # 如果还是失败，尝试提取可能的 JSON 片段
        extracted = extract_json_from_text(text)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
                
    return fallback


def fix_json_text(text: str) -> str:
    """
    修复常见的 JSON 格式问题
    
    Args:
        text: 待修复的 JSON 文本
        
    Returns:
        修复后的文本
    """
    # 去除首尾空白
    text = text.strip()
    
    # 移除 ```json 和 ``` 包裹
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
        
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    # 修复缺失的引号
    # 简单修复：确保键被引号包裹（这是个简化版本）
    # 注意：这不是完美的，但处理常见情况
    
    # 修复尾随逗号
    # 移除对象或数组末尾的逗号
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # 修复单引号问题
    text = text.replace("'", '"')
    
    return text


def extract_json_from_text(text: str) -> Optional[str]:
    """
    从文本中尝试提取 JSON 片段
    
    Args:
        text: 包含 JSON 的文本
        
    Returns:
        提取出的 JSON 字符串，或 None
    """
    # 查找对象模式
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        return obj_match.group(0)
        
    # 查找数组模式
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        return arr_match.group(0)
        
    return None


def safe_str(obj: Any) -> str:
    """
    安全转换为字符串
    
    Args:
        obj: 任意对象
        
    Returns:
        字符串表示
    """
    if obj is None:
        return ""
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        人类可读的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """
    验证配置是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需键列表
        
    Returns:
        缺失的键列表
    """
    missing = []
    for key in required_keys:
        if key not in config or config[key] is None:
            missing.append(key)
    return missing


def clean_filename(filename: str) -> str:
    """
    清理文件名中的非法字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        清理后的文件名
    """
    # 移除或替换非法字符
    illegal_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(illegal_chars, '_', filename)
    # 去除首尾点和空格
    cleaned = cleaned.strip('. ')
    # 确保不为空
    if not cleaned:
        cleaned = "unnamed_file"
    return cleaned


def has_chinese(text: str) -> bool:
    """
    检查文本是否包含中文字符
    
    Args:
        text: 待检查的文本
        
    Returns:
        是否包含中文
    """
    return any('\u4e00' <= c <= '\u9fff' for c in text)
