"""
向量存储模块
"""
from .faiss_store import FAISSVectorStore, get_faiss_vector_store

__all__ = ["FAISSVectorStore", "get_faiss_vector_store"]
