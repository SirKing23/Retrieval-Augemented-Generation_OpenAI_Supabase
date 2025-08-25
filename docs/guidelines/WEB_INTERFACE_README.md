# 🤖 RAG System Web Interface

A modern, interactive web interface for your Retrieval-Augmented Generation (RAG) system. Ask questions about your documents through an intuitive chat interface with real-time performance monitoring.

![RAG System Interface](https://img.shields.io/badge/Status-Ready-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)

## ✨ Features

### 🎯 Core Functionality
- **Interactive Chat Interface** - Natural conversation with your documents
- **Intelligent Document Search** - AI-powered semantic search
- **Source Citations** - See exactly where answers come from
- **Real-time Performance Monitoring** - Track cache hits, API calls, and response times

### 🚀 Performance Optimizations
- **Persistent Caching** - Embedding and response caching for faster responses
- **Batch Processing** - Efficient document processing
- **Memory Management** - Optimized for long-running sessions
- **API Call Tracking** - Monitor and optimize OpenAI API usage

### 📱 User Experience
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Real-time Stats** - Live performance metrics
- **Document Processing** - Easy upload and processing interface
- **WebSocket Support** - Real-time chat functionality

## 🏃‍♂️ Quick Start

### Option 1: Easy Launch (Recommended)
```bash
# For Windows users
start_website.bat

# For all users
python start_website.py
```

### Option 2: Manual Launch
```bash
# Install dependencies
pip install -r web_requirements.txt

# Start the server
uvicorn rag_webapp:app --host 0.0.0.0 --port 8000 --reload
```

## 📋 Prerequisites

### 1. Environment Variables
Create a `.env` file in your project directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Optional customizations
MAIN_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
MAX_TOKEN_RESPONSE=700
VECTOR_SEARCH_THRESHOLD=0.5
VECTOR_SEARCH_MATCH_COUNT=3
```

### 2. Documents Folder
Place your documents in the `uploaded_docs/` folder:
- **Supported formats**: PDF, TXT, DOCX
- **Best practices**: Use clear, well-structured documents
- **Processing**: Use the web interface to process new documents

### 3. Supabase Database
Make sure your Supabase database has the required table and function:

```sql
-- Create the document embeddings table
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    file_path TEXT,
    chunk_index INTEGER,
    filename TEXT,
    file_type TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create the vector similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 3
)
RETURNS TABLE (
    id INTEGER,
    content TEXT,
    metadata JSONB,
    file_path TEXT,
    chunk_index INTEGER,
    filename TEXT,
    file_type TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        document_embeddings.id,
        document_embeddings.content,
        document_embeddings.metadata,
        document_embeddings.file_path,
        document_embeddings.chunk_index,
        document_embeddings.filename,
        document_embeddings.file_type,
        1 - (document_embeddings.embedding <=> query_embedding) AS similarity
    FROM document_embeddings
    WHERE 1 - (document_embeddings.embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
```

## 🌐 Accessing the Interface

Once started, access the web interface at:

- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Alternative API Docs**: http://localhost:8000/api/redoc

## 📱 Interface Overview

### Chat Section
- **Message Input**: Type your questions about the documents
- **Response Display**: AI-generated answers with source citations
- **Source Information**: See which documents and pages provided the information
- **Performance Metrics**: Response time, cache status, and document count

### Performance Dashboard
- **Cache Statistics**: Hit rates for embedding and response caches
- **API Call Tracking**: Monitor OpenAI API usage and savings
- **System Health**: Real-time system status monitoring
- **Session Analytics**: Track questions asked and response times

### Document Management
- **Processing Interface**: Process new documents from the uploaded_docs folder
- **Status Monitoring**: Track processing progress and results
- **File Statistics**: See how many documents are indexed

## 🛠️ API Endpoints

### Core Endpoints
- `POST /api/ask` - Ask a question
- `POST /api/upload` - Process documents
- `GET /api/stats` - Get performance statistics
- `GET /api/health` - Health check
- `DELETE /api/cache` - Clear caches

### WebSocket
- `WS /ws/{session_id}` - Real-time chat connection

### Example API Usage

```python
import requests

# Ask a question
response = requests.post('http://localhost:8000/api/ask', json={
    'question': 'What is ProductCenter?',
    'session_id': 'my-session'
})

data = response.json()
print(f"Answer: {data['response']}")
print(f"Sources: {len(data['sources'])} documents")
print(f"Response time: {data['response_time']:.2f}s")
```

## 🎨 Customization

### Styling
The interface uses modern CSS with:
- **Responsive design** for all screen sizes
- **Dark/light theme** compatibility
- **Smooth animations** and transitions
- **Professional color scheme**

### Configuration
Modify these settings in your `.env` file:
- **Model Selection**: Choose between GPT models
- **Response Length**: Adjust max tokens
- **Search Parameters**: Tune similarity thresholds
- **Cache Settings**: Configure cache sizes

## 🔧 Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check if port 8000 is available
netstat -an | grep 8000

# Use a different port
uvicorn rag_webapp:app --port 8080
```

**Missing dependencies:**
```bash
# Install all requirements
pip install -r web_requirements.txt

# Check specific package
pip show fastapi
```

**Environment variables not loading:**
```bash
# Check if .env file exists
ls -la .env

# Test environment variable
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

**Database connection issues:**
- Verify Supabase URL and key
- Check database table structure
- Ensure vector extension is enabled

### Performance Optimization

**For better performance:**
1. **Use caching** - Let the system build up cache over time
2. **Batch processing** - Process multiple documents at once
3. **Optimize queries** - Use specific, focused questions
4. **Monitor stats** - Check the performance dashboard regularly

## 📊 Monitoring and Analytics

The interface provides comprehensive monitoring:

### Cache Performance
- **Hit Rates**: Percentage of requests served from cache
- **API Savings**: Number of OpenAI API calls saved
- **Cache Size**: Storage usage of embedding and response caches

### Response Analytics
- **Average Response Time**: Track system performance
- **Document Retrieval**: Number of relevant documents found
- **Session Tracking**: Monitor user interactions

### System Health
- **Memory Usage**: Track system resource consumption
- **Active Sessions**: Monitor concurrent users
- **Error Rates**: Identify and troubleshoot issues

## 🚀 Production Deployment

For production use:

1. **Security**: Set up proper authentication and HTTPS
2. **Scaling**: Use multiple workers with gunicorn
3. **Monitoring**: Set up proper logging and monitoring
4. **Backup**: Regular cache and database backups

```bash
# Production command
gunicorn rag_webapp:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📝 License

This project is part of the RAG System implementation. See the main project README for license information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Happy document chatting! 🤖📚**
