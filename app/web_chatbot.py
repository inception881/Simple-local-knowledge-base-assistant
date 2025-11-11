"""
Simplified CHATBOT Web Interface - Supporting single session conversations and document uploads
"""
import os
import sys
import uuid
import time
from pathlib import Path
import streamlit as st
from datetime import datetime

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.chains.faiss_conversational_chain import get_conversational_chain

# Configure page
st.set_page_config(
    page_title="Personal Knowledge Base Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style settings
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}

.user-message {
    background-color: #e6f7ff;
    border-left: 5px solid #1890ff;
    margin-left: 20%;
    margin-right: 2%;
}

.bot-message {
    background-color: #f0f0f0;
    border-left: 5px solid #52c41a;
    margin-right: 20%;
    margin-left: 2%;
}

.message-content {
    word-wrap: break-word;
}

.message-header {
    font-size: 0.8rem;
    color: #666;
    margin-bottom: 0.5rem;
}

.sources {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.8rem;
    padding-top: 0.5rem;
    border-top: 1px solid #ddd;
}

.source-item {
    margin-bottom: 0.3rem;
    padding-left: 1rem;
}

.sidebar-header {
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

.stButton>button {
    width: 100%;
}

.upload-section {
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px dashed #ddd;
    margin-top: 1rem;
}

.footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.8rem;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

def generate_session_id():
    """Generate a unique session ID"""
    return f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"

def init_session():
    """Initialize session state"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = get_conversational_chain(st.session_state.session_id)

def upload_document():
    """Upload document to knowledge base"""
    uploaded_file = st.session_state.uploaded_file
    
    try:
        # Load document and add to vector store
        st.session_state.chatbot.add_documents(uploaded_file)
        
        # Update session state
        st.session_state.upload_success = True
        st.session_state.upload_message = f"Successfully added document '{uploaded_file.name}' to knowledge base"
    
    except Exception as e:
        st.session_state.upload_success = False
        st.session_state.upload_message = f"Failed to add document: {str(e)}"

def main():
    """Main function"""
    # Initialize session
    init_session()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Personal Knowledge Base Assistant")
        st.markdown("---")
        
        # Document upload
        st.markdown("### 📚 Knowledge Base Management")
        
        with st.expander("Upload Documents to Knowledge Base"):
            st.file_uploader(
                "Select file to upload",
                type=["pdf", "txt", "docx", "md"],
                key="uploaded_file",
                on_change=upload_document
            )
            
            if "upload_success" in st.session_state:
                if st.session_state.upload_success:
                    st.success(st.session_state.upload_message)
                else:
                    st.error(st.session_state.upload_message)
        
        # System information
        st.markdown("---")
        st.markdown("### ⚙️ System Information")
        
        with st.expander("Model Configuration"):
            st.markdown(f"**LLM Model**: {Config.ANTHROPIC_MODEL_NAME}")
            st.markdown(f"**Embedding Model**: {Config.OPENAI_EMBEDDING_MODEL}")
            st.markdown(f"**Temperature**: {Config.TEMPERATURE}")
            st.markdown(f"**Retrieval Count**: Top {Config.TOP_K}")
    
    # Main interface
    st.header("💬 Intelligent Conversation Assistant")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
                    
                    # Display sources (if any)
                    metadata = message.get("metadata", {})
                    if "sources" in metadata and metadata["sources"]:
                        sources = metadata["sources"]
                        with st.expander("View Reference Sources"):
                            for i, source in enumerate(sources):
                                # Use title field as filename
                                title = source.get('title', 'Unknown Document')
                                content = source.get('content', '').strip()
                                st.markdown(f"**Source {i+1}**: {title}")
                                st.markdown(f"```\n{content[:200]}...\n```")
    
    # Chat input
    if prompt := st.chat_input("Enter your question..."):
        # Add user message to interface
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call conversation chain
                    response = st.session_state.chatbot.call(prompt)
                    
                    answer = response["answer"]
                    sources = response.get("sources", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("View Reference Sources"):
                            for i, source in enumerate(sources):
                                # Get metadata directly from source
                                metadata = source.get("metadata", {})
                                title = metadata.get("file_name", "Unknown Document")
                                content = source.get("content", "").strip()
                                
                                st.markdown(f"**Source {i+1}**: {title}")
                                st.markdown(f"```\n{content[:200]}...\n```")
                    
                    # Save assistant reply
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "sources": [
                                {
                                    "title": s.get("metadata", {}).get("file_name", "Unknown Document"),
                                    "content": s.get("content", "")
                                }
                                for s in sources
                            ]
                        }
                    }
                    st.session_state.messages.append(assistant_message)
                
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")
    
    # Footer
    st.markdown("""
    <div class="footer">
        Intelligent Conversation Assistant © 2025 | Built with LangChain and Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
