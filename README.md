# Retrieval-Augmented Generation (RAG) System

A  RAG system that combines OpenAI's language models with Supabase for vector storage, enabling intelligent document querying and knowledge extraction.

## 1. Installation Instructions

### Prerequisites
- Python 3.8 or higher
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SirKing23/Retrieval-Augemented-Generation_Local.git
   cd Retrieval-Augemented-Generation_Local
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env template` to `.env`
   - Fill in your API keys and configuration:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     SUPABASE_URL=your_supabase_url_here
     SUPABASE_KEY=your_supabase_anon_key_here
     DOCUMENTS_DIR=./data/Knowledge_Base_Files
     ```

5. **Run the system**
   ```bash
   # Start the web server
   python src/server/start_server.py
   
   # Or use the terminal interface
   python src/interfaces/terminal_interface/RAG_Terminal_Interface.py
   ```

## 2. Capabilities and How to Use

### Core Features

- **Document Processing**: Supports PDF, DOCX, and TXT file formats
- **Intelligent Chunking**: Automatically splits documents into optimal chunks for retrieval
- **Vector Storage**: Uses Supabase for persistent vector embeddings storage
- **Smart Caching**: Implements embedding and response caching for improved performance
- **Multiple Interfaces**: 
  - Web interface with real-time chat
  - Terminal/CLI interface
  - REST API endpoints

### How to Use

#### Web Interface
1. Start the server: `python src/server/start_server.py`
2. Open your browser to `http://localhost:8000`
3. Upload documents using the interface
4. Ask questions about your documents in the chat

#### Terminal Interface
1. Run: `python src/interfaces/terminal_interface/RAG_Terminal_Interface.py`
2. Follow the prompts to upload documents or ask questions

#### API Endpoints
- `POST /upload/` - Upload documents
- `POST /query/` - Query documents
- `GET /documents/` - List uploaded documents
- `DELETE /documents/{file_id}` - Delete specific documents
- `WebSocket /ws/` - Real-time chat interface

### Supported Operations

- **Document Upload**: Drag and drop or programmatic upload
- **Document Management**: View, delete, and organize your knowledge base
- **Intelligent Querying**: Ask natural language questions about your documents
- **Source Citation**: Get references to specific document sections
- **Batch Processing**: Process multiple documents simultaneously

## 3. Tools and Technologies Used

### Core AI & Language Processing
- **OpenAI API**: GPT models for text generation and embeddings
- **LangChain**: Document processing, text splitting, and chain management
  - `langchain-core`: Core functionality
  - `langchain-community`: Community integrations
  - `langchain-openai`: OpenAI integrations
  - `langchain-text-splitters`: Advanced text chunking

### Vector Database & Storage
- **Supabase**: Vector database for embeddings storage and similarity search
- **PostgreSQL**: Underlying database with pgvector extension

### Document Processing
- **PyPDF2**: PDF document parsing and text extraction
- **python-docx**: Microsoft Word document processing
- **tiktoken**: Token counting and text measurement

### Web Framework & API
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for serving the application
- **Pydantic**: Data validation and settings management
- **WebSockets**: Real-time communication support

### HTTP & Networking
- **requests**: Synchronous HTTP requests
- **aiohttp**: Asynchronous HTTP client/server
- **CORS Middleware**: Cross-origin resource sharing support

### Utilities & Configuration
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bars for long-running operations
- **asyncio**: Asynchronous programming support

### Development & Monitoring
- **logging**: Comprehensive logging system
- **caching**: Smart caching for embeddings and responses
- **error handling**: Robust error management and recovery

### System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web/Terminal  │    │   FastAPI Server │    │   RAG Core      │
│   Interface     │◄──►│   (REST/WS API)  │◄──►│   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌──────────────────┐             │
                       │   Document       │◄────────────┘
                       │   Processor      │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │   Supabase       │
                       │   Vector DB      │
                       └──────────────────┘
```

This RAG system provides a basic solution for document-based question and answer features including caching, error handling, and multiple interface options.
