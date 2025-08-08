# Ollama Integration Setup Guide

## Overview
The RAG system has been successfully converted to use Ollama for offline AI inference while maintaining Supabase as the online vector database.

## Changes Made

### 1. Model Configuration
- **Embedding Model**: `nomic-embed-text:latest` via Ollama
- **Chat Model**: `deepseek-r1:8b` via Ollama  
- **Vector Database**: Supabase (unchanged)

### 2. Key Files Modified
- `src/rag_core/RAG_Core.py`: Updated to use Ollama API endpoints
- Configuration now uses `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, and `CHAT_MODEL`

### 3. API Changes
- Removed OpenAI dependencies and API key requirements
- Added Ollama connection validation
- Updated embedding generation to use Ollama's `/api/embeddings` endpoint
- Updated chat completion to use Ollama's `/api/generate` endpoint

## Setup Instructions

### 1. Install and Start Ollama
```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Start Ollama server
ollama serve
```

### 2. Pull Required Models
```bash
# Pull the embedding model
ollama pull nomic-embed-text:latest

# Pull the chat model  
ollama pull deepseek-r1:8b
```

### 3. Verify Models are Available
```bash
# List available models
ollama list
```

### 4. Configure Environment Variables
Copy `.env.template` to `.env` and update with your values:

```bash
cp .env.template .env
```

Update the `.env` file with your Supabase credentials:
```
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

### 5. Test the Integration
Run the test script to verify everything is working:

```bash
python test_ollama_integration.py
```

## Usage Examples

### Basic RAG System Usage
```python
from src.rag_core.RAG_Core import RAGSystem

# Initialize the system (will automatically validate Ollama connection)
rag_system = RAGSystem()

# Process documents (same as before)
rag_system.initialize_files("./data/Knowledge_Base_Files")

# Ask questions (now uses Ollama models)
response = rag_system.answer_this("What is machine learning?")
print(response["response"])
```

### Check Ollama Status
```python
# Check if Ollama is running and models are available
if rag_system.validate_ollama_connection():
    print("✅ Ollama is ready!")
else:
    print("❌ Ollama connection failed")
```

### View Performance Statistics
```python
# Get Ollama-specific statistics
stats = rag_system.get_ollama_call_summary()
print(stats)
```

## Key Benefits

1. **Offline Operation**: No internet required for AI inference (only for Supabase vector storage)
2. **Cost Effective**: No API costs for embeddings or chat completion
3. **Privacy**: All AI processing happens locally
4. **Customizable**: Easy to swap models by changing environment variables
5. **Performance**: Caching system reduces redundant model calls

## Troubleshooting

### Common Issues

1. **"Ollama connection failed"**
   - Ensure Ollama is running: `ollama serve`
   - Check if port 11434 is available
   - Verify `OLLAMA_BASE_URL` in environment variables

2. **"Model not found"**
   - Pull the required models: `ollama pull model-name`
   - Check available models: `ollama list`
   - Verify model names in environment variables match exactly

3. **"Embedding generation failed"**
   - Ensure `nomic-embed-text:latest` is pulled and available
   - Check Ollama logs for errors
   - Try testing with a simple text first

4. **"Chat completion failed"**
   - Ensure `deepseek-r1:8b` is pulled and available
   - Check if the model supports the requested parameters
   - Verify prompt formatting

### Performance Tips

1. **Model Loading**: First call to each model may be slow as it loads into memory
2. **Batch Processing**: Embeddings are processed sequentially (Ollama limitation)
3. **Memory**: Ensure sufficient RAM for both models to run simultaneously
4. **Cache**: Enable caching to reduce redundant model calls

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `EMBEDDING_MODEL` | Model for text embeddings | `nomic-embed-text:latest` |
| `CHAT_MODEL` | Model for chat completion | `deepseek-r1:8b` |
| `SUPABASE_URL` | Supabase project URL | Required |
| `SUPABASE_KEY` | Supabase anon key | Required |
| `MAX_TOKEN_RESPONSE` | Max tokens for responses | `700` |
| `VECTOR_SEARCH_THRESHOLD` | Similarity threshold | `0.5` |
| `VECTOR_SEARCH_MATCH_COUNT` | Max search results | `3` |
| `CACHE_DIR` | Cache directory | `./cache` |

## Next Steps

1. Test the system with your specific use case
2. Adjust model parameters as needed
3. Monitor performance and cache efficiency
4. Consider upgrading to larger models for better quality if hardware allows
