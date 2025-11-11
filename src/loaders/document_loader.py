"""
Document Loader - Using LangChain document loaders to load and process various document formats

This module provides a unified document loading interface supporting multiple file formats including PDF, Word, text, HTML, and Markdown.
It handles document loading, chunking, metadata addition, and processing state tracking to avoid reprocessing the same document.
"""
from pathlib import Path
import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from src.config import Config
from src.utils.text_splitter import get_recursive_splitter

# Path to file recording processed document identifiers
PROCESSED_DOCS_RECORD = Config.PROCESSED_DOCS_RECORD

class DocumentLoaderService:
    """
    Document Loading Service
    
    Provides unified document loading, processing, and management functionality,
    supporting multiple file formats with deduplication and status tracking.
    """
    
    # Supported file types and their corresponding loaders
    SUPPORTED_LOADERS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".html": TextLoader,
        ".htm": TextLoader,
        ".md": TextLoader,
    }
    
    def __init__(self):
        """Initialize document loader service, set up text splitter"""
        self.text_splitter = get_recursive_splitter()
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document and process it into chunks
        
        Args:
            file_path: File path
        
        Returns:
            List of Document objects (chunked)
        
        Raises:
            ValueError: When file type is not supported
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        # Check if file type is supported
        if suffix not in self.SUPPORTED_LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}, supported types: {list(self.SUPPORTED_LOADERS.keys())}")
        
        # Get appropriate loader class
        loader_class = self.SUPPORTED_LOADERS[suffix]
        
        # Create loader instance
        loader_kwargs = {"encoding": "utf-8"} if loader_class == TextLoader else {}
        loader = loader_class(str(path), **loader_kwargs)
        
        # Load document
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source": str(path),
                "file_name": path.name,
                "file_type": suffix
            })
        
        # Process into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def _process_directory(self, directory: Path, skip_processed: bool) -> List[Document]:
        """
        Process all supported documents in a directory
        
        Args:
            directory: Directory path
            skip_processed: Whether to skip already processed documents
        
        Returns:
            List of Document objects (chunked)
        """
        all_chunks = []
        
        # Recursively find all supported files
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_LOADERS:
                file_chunks = self._process_file(file_path, skip_processed)
                all_chunks.extend(file_chunks)
        
        return all_chunks
    
    def _process_file(self, file_path: Path, skip_processed: bool) -> List[Document]:
        """
        Process a single file
        
        Args:
            file_path: File path
            skip_processed: Whether to skip already processed documents
        
        Returns:
            List of Document objects (chunked)
        """
        # Check if document has already been processed
        doc_id = str(file_path)
        if skip_processed and self._is_document_processed(doc_id):
            print(f"Skipping already processed file: {file_path.name}")
            return []
        
        print(f"Loading file: {file_path.name}")
        try:
            # Load and process document
            chunks = self.load_document(str(file_path))
            
            # Record processed document
            self._record_processed_document(doc_id)
            print(f"  ✓ Success: {len(chunks)} chunks")
            
            return chunks
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return []
    
    def _is_document_processed(self, doc_id: str) -> bool:
        """
        Check if document has already been processed (to avoid duplicates)
        
        Args:
            doc_id: Document identifier (typically file path)
        
        Returns:
            Whether it has been processed
        """
        if not PROCESSED_DOCS_RECORD.exists():
            return False
        with open(PROCESSED_DOCS_RECORD, "r", encoding="utf-8") as f:
            processed_ids = f.read().splitlines()
        return doc_id in processed_ids
    
    def _record_processed_document(self, doc_id: str) -> None:
        """
        Record processed document identifier
        
        Args:
            doc_id: Document identifier (typically file path)
        """
        with open(PROCESSED_DOCS_RECORD, "a", encoding="utf-8") as f:
            f.write(f"{doc_id}\n")
    
    def batch_process_documents(self, documents: List[Document], batch_size: int = 10) -> List[List[Document]]:
        """
        Process documents in batches for large-scale document processing
        
        Args:
            documents: List of documents
            batch_size: Number of documents per batch
            
        Returns:
            List of batches, each containing multiple documents
        """
        batches = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batches.append(batch)
            print(f"Created batch {len(batches)}: {len(batch)} documents")
        
        return batches
        
    def save_uploaded_file(self, uploaded_file) -> Path:
        """
        Save uploaded file to documents directory
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Saved file path
        """
        if uploaded_file is None:
            raise ValueError("No upload file provided")
            
        # Ensure documents directory exists
        Config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create target file path
        file_path = Config.DOCUMENTS_DIR / uploaded_file.name
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        print(f"✓ Uploaded file saved to: {file_path}")
        return file_path

# Global singleton instance
_LOADER_INSTANCE = None

def get_document_loader() -> DocumentLoaderService:
    """
    Get document loader singleton instance
    
    Returns:
        DocumentLoaderService instance
    """
    global _LOADER_INSTANCE
    if _LOADER_INSTANCE is None:
        _LOADER_INSTANCE = DocumentLoaderService()
    return _LOADER_INSTANCE

def is_document_processed(doc_id: str) -> bool:
    """
    Convenience function to check if document has been processed
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Whether it has been processed
    """
    loader = get_document_loader()
    return loader._is_document_processed(doc_id)

def record_processed_document(doc_id: str) -> None:
    """
    Convenience function to record processed document
    
    Args:
        doc_id: Document identifier
    """
    loader = get_document_loader()
    loader._record_processed_document(doc_id)
