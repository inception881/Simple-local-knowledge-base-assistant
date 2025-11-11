"""
Embedding 模型管理
"""
from langchain_openai import OpenAIEmbeddings
from typing import Optional
from src.config import Config

def get_embeddings(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    获取 Embedding 模型实例
    
    Args:
        model: 模型名称，默认使用配置中的模型
        api_key: API 密钥，默认使用配置中的密钥
        base_url: API 基础 URL，默认使用配置中的 URL
        
    Returns:
        OpenAIEmbeddings 实例
    """
    return OpenAIEmbeddings(
        model=model or Config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=api_key or Config.OPENAI_API_KEY,
        base_url=base_url or Config.OPENAI_BASE_URL
    )

# 全局单例
_embeddings = None

def get_embeddings_singleton() -> OpenAIEmbeddings:
    """
    获取 Embedding 模型单例
    
    Returns:
        OpenAIEmbeddings 单例实例
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings
