"""
文本分割工具
"""
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from src.config import Config

def get_recursive_splitter():
    """
    获取递归字符分割器
    适用于大多数文本，会尝试在段落、句子等自然边界分割
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )

def get_character_splitter():
    """
    获取简单字符分割器
    适用于简单文本，按字符数分割
    """
    return CharacterTextSplitter(
        separator="\n",
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
    )

def get_token_splitter(encoding_name="cl100k_base"):
    """
    获取基于 Token 的分割器
    适用于需要精确控制 Token 数量的场景
    
    Args:
        encoding_name: 编码名称，默认为 OpenAI 的 cl100k_base
    """
    return TokenTextSplitter(
        encoding_name=encoding_name,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

def split_text(text, splitter_type="recursive"):
    """
    分割文本
    
    Args:
        text: 要分割的文本
        splitter_type: 分割器类型，可选 "recursive", "character", "token"
    
    Returns:
        分割后的文本块列表
    """
    if splitter_type == "recursive":
        splitter = get_recursive_splitter()
    elif splitter_type == "character":
        splitter = get_character_splitter()
    elif splitter_type == "token":
        splitter = get_token_splitter()
    else:
        raise ValueError(f"不支持的分割器类型: {splitter_type}")
    
    return splitter.split_text(text)
