"""
FAISS 向量存储管理

该模块提供了对 FAISS 向量存储的统一管理接口，包括创建、加载、更新和查询等功能。
FAISS 是一个高效的向量相似度搜索库，用于存储文档的向量表示并进行快速检索。
"""
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.config import Config
from src.embedding import get_embeddings, get_embeddings_singleton
from src.loaders.document_loader import get_document_loader

# 向量库存储路径
FAISS_INDEX_PATH = Config.FAISS_INDEX_PATH

# 确保目录存在
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

class FAISSVectorStore:
    """FAISS 向量存储管理类"""
    
    def __init__(self, embeddings: Optional[Embeddings] = None):
        """
        初始化 FAISS 向量存储管理器
        
        Args:
            embeddings: 嵌入模型，默认使用全局单例
        """
        self.embeddings = embeddings or get_embeddings_singleton()
        self.vector_store = self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self) -> FAISS:
        """
        加载或创建 FAISS 向量存储
        
        如果存在已保存的向量库，则加载；否则创建一个空的向量库。
        
        Returns:
            FAISS 向量存储实例
        """
        # 检查是否存在已保存的向量库
        if FAISS_INDEX_PATH.exists() and any(FAISS_INDEX_PATH.iterdir()):
            try:
                print(f"✅ 加载已有向量库（路径：{FAISS_INDEX_PATH}）")
                vector_store = FAISS.load_local(
                    folder_path=str(FAISS_INDEX_PATH),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return vector_store
            except Exception as e:
                print(f"⚠️ 加载向量库失败: {e}，将创建新的向量库")
        
        # 如果没有文档，创建一个空的向量存储
        print("⚠️ 未找到已有向量库或加载失败，创建空的向量存储")
        vector_store = FAISS.from_texts(
            ["初始化文档"], self.embeddings
        )
        vector_store.save_local(str(FAISS_INDEX_PATH))
        return vector_store
    
    def get_retriever(self, k: int = None):
        """
        获取检索器
        
        Args:
            k: 检索的文档数量，默认使用配置中的值
            
        Returns:
            检索器实例
        """
        search_kwargs = {"k": k or Config.TOP_K}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def add_documents(self, documents: List[Document], batch_size: int = 10, ids: Optional[List[str]] = None) -> bool:
        """
        添加文档到向量存储
        
        Args:
            documents: 要添加的文档chunk列表
            batch_size: 批处理大小
            ids: 可选的文档ID列表，如果提供，长度必须与documents相同
            
        Returns:
            是否成功添加文档
        """
        if not documents:
            print("⚠️ 无文档可添加")
            return False
        
        # 生成文档ID
        if ids is None:
            # 为每个文档生成一个ID，格式为：源文件路径_UUID
            generated_ids = []
            for doc in documents:
                source = doc.metadata.get("source", "")
                doc_uuid = str(uuid.uuid4())
                generated_ids.append(f"{source}_{doc_uuid}")
            ids = generated_ids
        
        # 确保文档和ID数量一致
        if len(documents) != len(ids):
            raise ValueError(f"文档数量({len(documents)})与ID数量({len(ids)})不匹配")
        
        # 使用文档加载器服务的批处理功能
        loader = get_document_loader()
        batches = loader.batch_process_documents(documents, batch_size)
        id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
        print(f"✅ 文档分批完成：共 {len(batches)} 个批次")
        
        # 向量化并合并到向量库（使用预先分好的批次）
        for i, (batch, batch_ids) in enumerate(zip(batches, id_batches)):
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(batch, self.embeddings, ids=batch_ids)
            else:
                # 使用add_documents而不是from_documents和merge_from，以便传递IDs
                self.vector_store.add_documents(documents=batch, ids=batch_ids)
            print(f"✅ 已处理批次 {i+1}/{len(batches)}")
        
        # 保存更新后的向量库到本地
        if self.vector_store:
            self.vector_store.save_local(str(FAISS_INDEX_PATH))
            print(f"✅ 向量库更新完成，已保存到：{FAISS_INDEX_PATH}")
            return True
        
        return False
    

    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认使用配置中的值
            
        Returns:
            相关文档列表
        """
        return self.vector_store.similarity_search(query, k=k or Config.TOP_K)
    
    def search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """
        搜索相关文档并返回相似度分数
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认使用配置中的值
            
        Returns:
            (文档, 分数) 元组列表
        """
        return self.vector_store.similarity_search_with_score(query, k=k or Config.TOP_K)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存向量库到本地
        
        Args:
            path: 保存路径，默认使用配置中的路径
        """
        save_path = path or str(FAISS_INDEX_PATH)
        self.vector_store.save_local(save_path)
        print(f"✅ 向量库已保存到：{save_path}")
    
    # def load_documents_and_update(self, document_paths: List[str]) -> bool:
    #     """
    #     加载文档并更新向量库
        
    #     Args:
    #         document_paths: 文档路径列表
            
    #     Returns:
    #         是否成功更新
    #     """
    #     print(f"\n📚 开始更新知识库（新文档数：{len(document_paths)}）")
        
    #     # 使用文档加载器服务加载文档
    #     loader = get_document_loader()
    #     all_docs = loader.process_documents(document_paths, skip_processed=True)
        
    #     if not all_docs:
    #         print("⚠️ 无新增文档，知识库未更新")
    #         return False
        
    #     print(f"✅ 成功加载 {len(all_docs)} 个新文档")
        
    #     # 添加文档到向量库
    #     return self.add_documents(all_docs)
    
    def delete(self, ids: List[str]) -> bool:
        """
        从向量库中删除指定ID的文档
        
        Args:
            ids: 要删除的文档ID列表
            
        Returns:
            是否成功删除
        """
        if not ids:
            print("⚠️ 未提供要删除的文档ID")
            return False
        
        try:
            self.vector_store.delete(ids=ids)
            # 保存更新后的向量库到本地
            self.save()
            print(f"✅ 成功删除 {len(ids)} 个文档")
            return True
        except Exception as e:
            print(f"⚠️ 删除文档失败: {e}")
            return False
    
    def delete_by_source(self, source_paths: List[str]) -> bool:
        """
        根据源文件路径删除文档
        
        删除所有ID以指定源文件路径开头的文档（忽略UUID部分）
        
        Args:
            source_paths: 源文件路径列表
            
        Returns:
            是否成功删除
        """
        if not source_paths:
            print("⚠️ 未提供要删除的源文件路径")
            return False
        
        try:
            # 获取所有文档ID
            all_ids = list(self.vector_store.index_to_docstore_id.values())
            
            # 找出匹配的ID
            ids_to_delete = []
            for source_path in source_paths:
                for doc_id in all_ids:
                    # 检查ID是否以源文件路径开头（格式为source_path_uuid）
                    if doc_id.startswith(f"{source_path}_"):
                        ids_to_delete.append(doc_id)
            
            if not ids_to_delete:
                print("⚠️ 未找到匹配的文档")
                return False
            
            # 删除匹配的文档
            self.vector_store.delete(ids=ids_to_delete)
            
            # 保存更新后的向量库到本地
            self.save()
            print(f"✅ 成功删除 {len(ids_to_delete)} 个文档")
            return True
        except Exception as e:
            print(f"⚠️ 删除文档失败: {e}")
            return False
    
    def clear(self) -> None:
        """清空向量库"""
        # 创建一个新的空向量库
        self.vector_store = FAISS.from_texts(
            ["初始化文档"], self.embeddings
        )
        self.save()
        print("✅ 向量库已清空")


# 全局单例
_vector_store_instance = None

def get_faiss_vector_store() -> FAISSVectorStore:
    """
    获取 FAISS 向量存储单例
    
    Returns:
        FAISSVectorStore 单例实例
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = FAISSVectorStore()
    return _vector_store_instance
