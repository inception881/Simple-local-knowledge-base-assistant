"""
聊天模型管理
"""
from langchain_anthropic import ChatAnthropic
from typing import Optional
from src.config import Config

def get_chat_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> ChatAnthropic:
    """
    获取聊天模型实例
    
    Args:
        model: 模型名称，默认使用配置中的模型
        api_key: API 密钥，默认使用配置中的密钥
        base_url: API 基础 URL，默认使用配置中的 URL
        temperature: 温度参数，默认使用配置中的温度
        max_tokens: 最大生成 token 数，默认使用配置中的值
        
    Returns:
        ChatAnthropic 实例
    """
    # 创建ChatAnthropic实例
    chat_model = ChatAnthropic(
        model=model or Config.ANTHROPIC_MODEL_NAME,
        anthropic_api_key=api_key or Config.ANTHROPIC_API_KEY,
        base_url=base_url or Config.ANTHROPIC_BASE_URL,
        temperature=temperature if temperature is not None else Config.TEMPERATURE,
        max_tokens=max_tokens or Config.MAX_TOKENS,
    )
    
    return chat_model

# 全局单例
_chat_model = None

def get_chat_model_singleton() -> ChatAnthropic:
    """
    获取聊天模型单例
    
    Returns:
        ChatAnthropic 单例实例
    """
    global _chat_model
    if _chat_model is None:
        _chat_model = get_chat_model()
    return _chat_model
