# 🤖 Simple-local-knowledge-base-assistant


<div align="center">
  <img src="https://img.shields.io/badge/LangChain-0.1.4-blue" alt="LangChain">
  <img src="https://img.shields.io/badge/Python-3.9+-green" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Streamlit-1.31.0-red" alt="Streamlit">
</div>


**Simple-local-knowledge-base-assistant** is a powerful Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your documents. Upload PDFs, Word documents, or text files and ask questions in natural language to get accurate, context-aware responses based on the content of your documents.

## ✨ Features

- **📄 Multi-format Document Support**: PDF, Word, TXT, Markdown
- **🔍 Advanced Retrieval**: FAISS vector search for fast and accurate document retrieval
- **💬 Conversational Interface**: Natural dialogue with memory of previous exchanges
- **🧠 Context-Aware Responses**: Answers are grounded in your documents with source citations
- **🚀 Easy Deployment**: Simple setup with Streamlit web interface
- **🔒 Privacy-Focused**: Your documents stay on your machine



## 🛠️ Technology Stack

- **LLM**: Claude (Anthropic)
- **Vector Store**: FAISS
- **Embeddings**: OpenAI/Qwen Embeddings
- **Web Framework**: Streamlit
- **Document Processing**: LangChain document loaders

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- API keys for Claude and OpenAI Embeddings

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/simple-local-knowledge-base-assistant.git
cd knowledgegpt
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory:

```
ANTHROPIC_LLM_API_KEY=your_anthropic_api_key
OPENAI_EMBEDDING_API_KEY=your_openai_api_key
```

4. **Run the application**

```bash
streamlit run app/web_chatbot.py
```

5. **Upload documents and start chatting!**

## 📚 Usage

1. **Upload documents**: Click "Upload Documents to Knowledge Base" in the sidebar
2. **Ask questions**: Type your questions in the chat input
3. **View sources**: Expand the "View Reference Sources" section to see which parts of your documents were used

## 🧩 Project Structure

```

Simple-local-knowledge-base-assistant/
├── app/                          # Application layer
│   └── web_chatbot.py            # Streamlit web interface
├── src/                          # Core source code
│   ├── chains/                   # LangChain chains
│   ├── loaders/                  # Document loaders
│   ├── prompts/                  # Prompt templates
│   ├── utils/                    # Utility functions
│   ├── vectorstores/             # Vector store implementations
│   └── config.py                 # Configuration
├── data/                         # Data directory
│   ├── documents/                # Document storage
│   └── faiss_index/              # FAISS vector indices
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## 🔧 Configuration

You can adjust various settings in `src/config.py`:

- LLM parameters (model, temperature)
- Embedding model
- Retrieval parameters (TOP_K, similarity threshold)
- Chunking parameters (size, overlap)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [LangChain](https://python.langchain.com/) for the amazing RAG framework
- [Anthropic Claude](https://www.anthropic.com/claude) for the powerful LLM
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Streamlit](https://streamlit.io/) for the easy-to-use web framework

---

<p align="center">
  Made with ❤️ by Your Name
</p>
