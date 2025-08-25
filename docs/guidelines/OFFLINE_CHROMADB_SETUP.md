# RAG System - Offline ChromaDB Setup

This guide explains how to set up and use the RAG system with ChromaDB as an offline vector database instead of Supabase.

## 🌟 Benefits of Offline Setup

- ✅ **Complete Privacy**: All data stays on your local machine
- ✅ **No Internet Required**: Works entirely offline after initial setup
- ✅ **No Subscription Costs**: No cloud database fees
- ✅ **Faster Performance**: No network latency for vector searches
- ✅ **Data Control**: Full control over your data and embeddings

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements/requirements_optimized.txt
```

### 2. Run Migration Script

```bash
python scripts/migrate_to_offline.py
```

This script will:
- Install ChromaDB
- Create a `.env` configuration file
- Test connections to ChromaDB and Ollama
- Provide setup instructions

### 3. Configure Environment

Edit the `.env` file created by the migration script:

```env
# ChromaDB Configuration (replaces Supabase)
CHROMADB_PATH=./data/chromadb
COLLECTION_NAME=document_embeddings

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text:latest
CHAT_MODEL=deepseek-r1:8b

# AI Parameters
MAX_TOKEN_RESPONSE=700
VECTOR_SEARCH_THRESHOLD=0.7
VECTOR_SEARCH_MATCH_COUNT=5
MAX_CHUNK_TOKENS=600
OVERLAP_CHUNK_TOKENS=300

# Cache Configuration
CACHE_DIR=./cache

# Document Processing
DOCUMENTS_DIR=./data/Knowledge_Base_Files
PROCESSED_FILES_DIR=./data/processed_files.txt
```

### 4. Ensure Ollama is Running

Make sure Ollama is installed and running with the required models:

```bash
# Install models if not already installed
ollama pull nomic-embed-text:latest
ollama pull deepseek-r1:8b

# Verify Ollama is running
ollama list
```

## 📁 Directory Structure

After setup, your directory structure will look like:

```
Retrieval-Augemented-Generation/
├── data/
│   ├── chromadb/              # ChromaDB database files
│   │   ├── chroma.sqlite3     # SQLite database
│   │   └── ...                # Other ChromaDB files
│   ├── Knowledge_Base_Files/  # Your documents
│   └── processed_files.txt    # Tracking processed files
├── cache/                     # Embedding and response cache
├── src/rag_core/
│   └── RAG_Core.py           # Updated for ChromaDB
└── .env                      # Configuration file
```

## 🛠 Usage Examples

### Basic Usage

```python
from src.rag_core.RAG_Core import RAGSystem

# Initialize the offline RAG system
rag = RAGSystem()

# Process documents (one-time setup)
rag.initialize_files('./data/Knowledge_Base_Files')

# Ask questions
response = rag.answer_this('What is the main topic of the documents?')
print(response['response'])
print(f"Sources: {len(response['sources'])}")
```

### Advanced Usage

```python
# Get performance metrics
metrics = rag.get_performance_report()
print(f"Cache hit rate: {metrics['cache_performance']['overall_cache_efficiency']['cache_hit_rate']}%")

# Clear cache if needed
rag.clear_cache()

# Get detailed source information
response = rag.answer_this('Your question here')
for source in response['sources']:
    print(f"Source: {source['title']} (Relevance: {source['relevance_score']:.1%})")
```

## 🔧 Configuration Options

### ChromaDB Settings

- **CHROMADB_PATH**: Local directory for ChromaDB storage
- **COLLECTION_NAME**: Name of the vector collection

### Vector Search Parameters

- **VECTOR_SEARCH_THRESHOLD**: Minimum similarity score (0.0-1.0)
- **VECTOR_SEARCH_MATCH_COUNT**: Number of documents to retrieve

### Performance Tuning

- **MAX_CHUNK_TOKENS**: Maximum tokens per document chunk
- **OVERLAP_CHUNK_TOKENS**: Overlap between chunks
- **CACHE_DIR**: Directory for response and embedding cache

## 🔍 Troubleshooting

### ChromaDB Issues

1. **Permission errors**: Ensure write permissions in `CHROMADB_PATH`
2. **Database locked**: Close any other processes using ChromaDB
3. **Corrupted database**: Delete the ChromaDB folder and reprocess documents

### Ollama Issues

1. **Connection refused**: Ensure Ollama is running (`ollama serve`)
2. **Model not found**: Pull required models (`ollama pull model-name`)
3. **Slow performance**: Consider using smaller models

### Performance Issues

1. **Slow vector search**: Reduce `VECTOR_SEARCH_MATCH_COUNT`
2. **Memory issues**: Reduce `MAX_CHUNK_TOKENS`
3. **Cache not working**: Check `CACHE_DIR` permissions

## 📊 Monitoring and Maintenance

### Check System Status

```python
# Check ChromaDB status
collection_count = rag.collection.count()
print(f"Documents in database: {collection_count}")

# Check cache performance
cache_stats = rag.get_cache_stats()
print(f"Embedding cache hit rate: {cache_stats['embedding_hit_rate']}")
print(f"Response cache hit rate: {cache_stats['response_hit_rate']}")
```

### Database Maintenance

```python
# Delete specific file embeddings
rag.delete_file_embeddings('filename.pdf')

# Clear all caches
rag.clear_cache()

# Reset chat history
rag.reset_chat_history()
```

## 🆚 Comparison: Online vs Offline

| Feature | Supabase (Online) | ChromaDB (Offline) |
|---------|-------------------|-------------------|
| **Privacy** | Data in cloud | Data local |
| **Cost** | Subscription fees | Free |
| **Internet** | Required | Not required |
| **Performance** | Network dependent | Local speed |
| **Scalability** | Cloud managed | Hardware limited |
| **Setup** | Cloud configuration | Local installation |

## 🔐 Data Migration

If you have existing data in Supabase and want to migrate:

1. Export your data from Supabase
2. Reprocess documents with the new ChromaDB system
3. The system will automatically create new embeddings

Note: Direct migration of embeddings is not supported due to different storage formats.

## 🚨 Important Notes

- **Backup**: Regularly backup your `CHROMADB_PATH` directory
- **Portability**: ChromaDB files are portable across systems
- **Version**: Keep ChromaDB version consistent across environments
- **Models**: Ensure same Ollama models are available on all systems

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [RAG System Implementation Guide](./docs/guidelines/IMPLEMENTATION_GUIDE.md)

---

For questions or issues, please check the troubleshooting section or create an issue in the repository.
