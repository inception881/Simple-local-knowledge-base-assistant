"""
FAISS-based Conversational RAG Chain - Multi-turn dialogue with memory
Implemented using LangChain Agent
"""
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain.agents.middleware import SummarizationMiddleware

from typing import List, Dict, Any
from pathlib import Path

from src.config import Config
from src.prompts.templates import PromptTemplate
from src.chat_model import get_chat_model
from src.loaders.document_loader import get_document_loader
from src.vectorstores.faiss_store import get_faiss_vector_store

# Vector store path
FAISS_INDEX_PATH = Config.FAISS_INDEX_PATH

# Ensure directory exists
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

class RetrievalTool(BaseTool):
    """Tool for retrieving documents from vector store"""
    
    name: str = "retrieval_tool"
    description: str = "Retrieve relevant documents from knowledge base"
    retriever: Any = None  # Add retriever as Pydantic field
    last_docs: List[Document] = []  # Store the most recent retrieved documents
    
    def __init__(self, retriever):
        """Initialize retrieval tool"""
        super().__init__(retriever=retriever)  # Pass retriever via parameter
    
    def _run(self, query: str) -> str:
        """Execute retrieval operation"""
        docs = self.retriever.invoke(query)
        self.last_docs = docs  # Save retrieved documents
        
        if not docs:
            return "No relevant documents found."
        
        # Format documents
        formatted_docs = "\n\n".join([f"<doc>\n{doc.page_content}\n</doc>" for doc in docs])
        return formatted_docs
    
    def get_last_docs(self) -> List[Document]:
        """Get the most recently retrieved documents"""
        return self.last_docs

class FAISSConversationalRAGChain:
    """FAISS-based Conversational RAG Chain using Agent implementation"""
    
    def __init__(self, session_id: str = "default"):
        """
        Initialize
        
        Args:
            session_id: Session ID
        """
        self.session_id = session_id
        
        # LLM
        self.llm = get_chat_model()
        
        # Load documents from data/documents directory
        self.document_loader = get_document_loader()
        
        # Get FAISS vector store
        self.faiss_store = get_faiss_vector_store()
        self.retriever = self.faiss_store.get_retriever()
        
        # Memory - use simple list to store conversation history
        self.history = []
        
        # Get prompt
        self.prompt_template = PromptTemplate.template
        
        # Create retrieval tool
        self.retrieval_tool = RetrievalTool(self.retriever)
        
        # Create agent with middleware for summarization
        self.agent = create_agent(
            model=self.llm,
            tools=[self.retrieval_tool],
            middleware=[
                SummarizationMiddleware(
                    model=get_chat_model(model="claude-sonnet-4-5"),
                    max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
                    messages_to_keep=20,  # Keep last 20 messages after summary
                ),
            ],
        )
    
    def call(self, question: str) -> dict:
        """
        Call Agent
        
        Args:
            question: User question
        
        Returns:
            Dictionary containing answer and source documents
        """
        # Ensure history has at least one message to avoid having only system message
        if not self.history:
            # Add initialization message
            self.history.append({"role": "user", "content": "Initialize conversation"})
            self.history.append({"role": "assistant", "content": "I'm ready to answer your questions."})
        
        # Convert history and current question to messages format
        messages = [{"role": "system", "content": self.prompt_template}]
        for msg in self.history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Call Agent with messages format
        result = self.agent.invoke({"messages": messages})
        
        # Result is a dictionary containing messages, last message is AI's answer
        if isinstance(result, dict) and "messages" in result:
            # Get all AI messages
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            
            # Get content of last AI message as answer
            if ai_messages:
                last_ai_message = ai_messages[-1]
                answer = last_ai_message.content
                if isinstance(answer, list):
                    # If content is a list, extract text parts
                    text_parts = [part.get("text", "") for part in answer if isinstance(part, dict) and "text" in part]
                    answer = " ".join(text_parts)
            else:
                answer = "Unable to generate an answer."
        else:
            # Try other ways to extract answer
            answer = result.get("answer", str(result))
        
        # Get documents used by retrieval tool
        source_docs = self.retrieval_tool.get_last_docs()
        
        # If tool didn't retrieve documents, use retriever directly
        if not source_docs:
            source_docs = self.retriever.invoke(question)
        
        # Save conversation history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        
        # If history is too long, keep only recent conversations
        if len(self.history) > Config.MAX_HISTORY_LENGTH * 2:
            self.history = self.history[-Config.MAX_HISTORY_LENGTH * 2:]
        
        print(f"self.history: {self.history}")
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
        }
    
    def add_documents(self, file_path: Path):
        """
        Add documents to vector store
        
        Args:
            file_path: File path
        """
        file_path = self.document_loader.save_uploaded_file(file_path)
        # Convert string path to Path object
        path = Path(file_path)
        
        # Process file
        chunks = self.document_loader._process_file(path, skip_processed=True)
        
        if not chunks:
            print(f"⚠️ File {path.name} did not generate any document chunks after processing")
            return
        
        # Use FAISS vector store service to add documents
        self.faiss_store.add_documents(chunks)
        
        # Update retriever
        self.retriever = self.faiss_store.get_retriever()
    
    def clear_memory(self):
        """Clear memory"""
        self.history = []
    
    def get_history(self) -> list:
        """Get conversation history - convert to LangChain message objects"""
        messages = []
        for msg in self.history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        
        return messages

# Session management
_sessions = {}

def get_conversational_chain(session_id: str) -> FAISSConversationalRAGChain:
    """Get or create conversation Chain"""
    if session_id not in _sessions:
        _sessions[session_id] = FAISSConversationalRAGChain(session_id)
    return _sessions[session_id]

def clear_session(session_id: str):
    """Clear session"""
    if session_id in _sessions:
        _sessions[session_id].clear_memory()
        del _sessions[session_id]
