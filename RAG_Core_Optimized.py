"""
Optimized RAG System with comprehensive improvements including:
- Performance optimizations
- Memory management
- Async operations
- Logging and monitoring
- Error recovery
- Security improvements
- Configuration management
"""

from openai import OpenAI
import os
from supabase import create_client, Client
from PyPDF2 import PdfReader
import docx
import json
from tqdm import tqdm
import tiktoken
import requests
import asyncio
import aiohttp
import time
import hashlib
import logging
import re
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache, wraps
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.schema import Document

# Load environment variables
load_dotenv()

@dataclass
class PerformanceMetrics:
    """Data class to track performance metrics"""
    embedding_time: float = 0.0
    search_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    num_documents_retrieved: int = 0
    query_length: int = 0

class RAGSystemConfig:
    """Configuration management class"""
    def __init__(self):
        self.ai_api_key = os.getenv("OPENAI_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.main_model = os.getenv("MAIN_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.max_token_response = int(os.getenv("MAX_TOKEN_RESPONSE", "700"))
        self.vector_search_threshold = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.5"))
        self.vector_search_match_count = int(os.getenv("VECTOR_SEARCH_MATCH_COUNT", "3"))
        self.max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", "600"))
        self.overlap_chunk_tokens = int(os.getenv("OVERLAP_CHUNK_TOKENS", "300"))
        
    def validate_config(self):
        """Validate all configuration parameters"""
        required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

class RAGLogger:
    """Structured logging class for the RAG system"""
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RAGSystem')
    
    def log_query(self, question: str, response_time: float, num_docs: int):
        """Log query information"""
        self.logger.info(
            f"Query processed: {len(question)} chars, "
            f"{response_time:.2f}s, {num_docs} docs retrieved"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error information"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        self.logger.info(
            f"Performance - Embedding: {metrics.embedding_time:.2f}s, "
            f"Search: {metrics.search_time:.2f}s, "
            f"Generation: {metrics.generation_time:.2f}s, "
            f"Total: {metrics.total_time:.2f}s"
        )

def measure_performance(func):
    """Decorator to measure method performance"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        if hasattr(self, 'logger'):
            self.logger.logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        return wrapper
    return decorator

class PersistentCacheManager:
    """Manages persistent cache storage for embeddings and responses"""
    
    def __init__(self, cache_dir: str = "rag_cache", max_cache_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Cache file paths
        self.embedding_cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
        self.response_cache_file = os.path.join(cache_dir, "response_cache.pkl")
        self.cache_metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup logging first
        self.logger = logging.getLogger('PersistentCache')
        
        # Load existing caches
        self.embedding_cache = self._load_cache(self.embedding_cache_file)
        self.response_cache = self._load_cache(self.response_cache_file)
    
    def _load_cache(self, cache_file: str) -> Dict[str, Any]:
        """Load cache from disk"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                cache_name = os.path.basename(cache_file).replace('.pkl', '')
                self.logger.info(f"Loaded {cache_name} with {len(cache)} items")
                return cache
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_file}: {e}")
                return {}
        return {}
    
    def _save_cache(self, cache: Dict[str, Any], cache_file: str) -> bool:
        """Save cache to disk"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save cache {cache_file}: {e}")
            return False
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            embedding_size = os.path.getsize(self.embedding_cache_file) if os.path.exists(self.embedding_cache_file) else 0
            response_size = os.path.getsize(self.response_cache_file) if os.path.exists(self.response_cache_file) else 0
            
            metadata = {
                "last_saved": datetime.now().isoformat(),
                "embedding_cache_size": len(self.embedding_cache),
                "response_cache_size": len(self.response_cache),
                "embedding_file_size_bytes": embedding_size,
                "response_file_size_bytes": response_size,
                "total_size_bytes": embedding_size + response_size
            }
            
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        return self.embedding_cache.get(text_hash)
    
    def set_embedding(self, text_hash: str, embedding: List[float]) -> bool:
        """Set embedding in cache"""
        self.embedding_cache[text_hash] = embedding
        return True
    
    def get_response(self, question_hash: str) -> Optional[Dict[str, Any]]:
        """Get response from cache"""
        return self.response_cache.get(question_hash)
    
    def set_response(self, question_hash: str, response_data: Dict[str, Any]) -> bool:
        """Set response in cache"""
        # Store response with timestamp
        cache_entry = {
            "response": response_data,
            "timestamp": datetime.now().isoformat(),
            "cached": True
        }
        self.response_cache[question_hash] = cache_entry
        return True
    
    def _cleanup_old_entries(self):
        """Remove old cache entries to manage size"""
        # Simple strategy: remove 20% of oldest entries from each cache
        for cache in [self.embedding_cache, self.response_cache]:
            if len(cache) > 1000:  # Limit number of items
                items_to_remove = len(cache) // 5
                keys_to_remove = list(cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    del cache[key]
                
                self.logger.info(f"Cleaned up {items_to_remove} old cache entries")
    
    def save_to_disk(self) -> bool:
        """Save all caches to disk"""
        try:
            # Cleanup old entries if needed
            self._cleanup_old_entries()
            
            # Save both caches
            embedding_success = self._save_cache(self.embedding_cache, self.embedding_cache_file)
            response_success = self._save_cache(self.response_cache, self.response_cache_file)
            
            # Save metadata
            self._save_metadata()
            
            if embedding_success and response_success:
                self.logger.info(f"Saved caches: {len(self.embedding_cache)} embeddings, {len(self.response_cache)} responses")
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Failed to save caches: {e}")
            return False
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.embedding_cache.clear()
        self.response_cache.clear()
        
        # Remove files
        for file_path in [self.embedding_cache_file, self.response_cache_file, self.cache_metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        embedding_size = os.path.getsize(self.embedding_cache_file) if os.path.exists(self.embedding_cache_file) else 0
        response_size = os.path.getsize(self.response_cache_file) if os.path.exists(self.response_cache_file) else 0
        
        return {
            "embedding_cache_items": len(self.embedding_cache),
            "response_cache_items": len(self.response_cache),
            "embedding_cache_file_size_bytes": embedding_size,
            "response_cache_file_size_bytes": response_size,
            "total_cache_size_bytes": embedding_size + response_size,
            "total_cache_size_mb": (embedding_size + response_size) / (1024 * 1024),
            "cache_directory": self.cache_dir
        }

class OptimizedRAGSystem:
    """
    Optimized RAG System with comprehensive improvements including:
    - Performance optimizations (caching, batch processing)
    - Memory management
    - Async operations
    - Logging and monitoring
    - Error recovery with retry logic
    - Security improvements
    - Configuration management
    """

    def __init__(self, config: Optional[RAGSystemConfig] = None, use_async: bool = False):
        """
        Initialize the optimized RAG system
        
        Args:
            config: Configuration object. If None, will create from environment variables
            use_async: Whether to enable async operations
        """
        # Initialize configuration
        self.config = config or RAGSystemConfig()
        self.config.validate_config()
        
        # Initialize logger
        self.logger = RAGLogger()
        
        # Initialize performance tracking
        self.performance_metrics = PerformanceMetrics()
        
        # Validate inputs
        self._validate_initialization_parameters()
        
        # AI parameters
        self.ai_api_key = self.config.ai_api_key
        self.ai_main_model = self.config.main_model
        self.ai_embedding_model = self.config.embedding_model
        self.ai_max_token_response = self.config.max_token_response
        self.ai_instance = OpenAI(api_key=self.config.ai_api_key)
        self.ai_default_no_response = "I couldn't find any relevant information in the knowledge base regarding your question."
        self.ai_system_role_prompt = "You are a helpful assistant with access to relevant documents and chat history. If the relevant documents and chat history has nothing to do with the user query, respond politely that the query is out of context."
        
        # Chunk parameters
        self.max_chunk_tokens = self.config.max_chunk_tokens
        self.overlap_chunk_tokens = self.config.overlap_chunk_tokens
        
        # Supabase parameters
        self.supabase_key = self.config.supabase_key
        self.supabase_url = self.config.supabase_url
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Supabase: {e}")
        
        self.vector_search_threshold = self.config.vector_search_threshold
        self.vector_search_match_count = self.config.vector_search_match_count
        
        # Files processing parameters
        self.processed_files_list = "processed_files.txt"
        
        # Chat History parameters with improved management
        self.chat_history = []
        self.is_initial_session = False
        self.max_history_length = 20
        
        # Async session for async operations
        self.use_async = use_async
        self.session = None
        
        # Initialize persistent cache manager
        self.cache_manager = PersistentCacheManager(cache_dir="rag_cache", max_cache_size_mb=100)
        
        # Keep reference to embedding cache for compatibility
        self.embedding_cache = self.cache_manager.embedding_cache
        
        # Statistics for cache performance and API calls
        self.cache_stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "response_hits": 0,
            "response_misses": 0
        }
        
        # API call tracking
        self.api_call_stats = {
            "openai_embedding_calls": 0,
            "openai_chat_calls": 0,
            "total_openai_calls": 0,
            "total_tokens_used": 0,
            "embedding_tokens_used": 0,
            "chat_tokens_used": 0
        }
        
        self.logger.logger.info("OptimizedRAGSystem initialized successfully with persistent caching")

    def _validate_initialization_parameters(self):
        """Validate initialization parameters"""
        if not self.config.ai_api_key or not isinstance(self.config.ai_api_key, str):
            raise ValueError("AI_API_Key must be a non-empty string")
        if not self.config.supabase_key or not isinstance(self.config.supabase_key, str):
            raise ValueError("Supabase_API_Key must be a non-empty string")
        if not self.config.supabase_url or not isinstance(self.config.supabase_url, str):
            raise ValueError("Supabase_URL must be a non-empty string")
        if self.config.max_token_response <= 0:
            raise ValueError("max_token_response must be positive")
        if not (0.0 <= self.config.vector_search_threshold <= 1.0):
            raise ValueError("Vector_Search_Threshold must be between 0.0 and 1.0")
        if self.config.vector_search_match_count <= 0:
            raise ValueError("Vector_Search_Match_Count must be positive")
        if self.config.max_chunk_tokens <= self.config.overlap_chunk_tokens:
            raise ValueError("MAX_CHUNK_TOKENS must be greater than OVERLAP_CHUNK_TOKENS")

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and basic security"""
        if not api_key or len(api_key) < 20:
            return False
        # Add more validation logic as needed
        return True

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            return ""
        # Remove or escape potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        # Limit length to prevent excessive token usage
        return sanitized[:10000]  # Limit to 10k characters

    @measure_performance
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding with caching and retry logic"""
        if not text or not text.strip():
            return None
            
        # Check persistent cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_embedding = self.cache_manager.get_embedding(text_hash)
        if cached_embedding is not None:
            self.cache_stats["embedding_hits"] += 1
            self.logger.logger.debug(f"Embedding cache HIT for text hash: {text_hash[:8]}...")
            return cached_embedding
        
        # Cache miss - call API
        self.cache_stats["embedding_misses"] += 1
        self.logger.logger.debug(f"Embedding cache MISS for text hash: {text_hash[:8]}...")
        
        try:
            start_time = time.time()
            response = self.ai_instance.embeddings.create(
                input=text,
                model=self.ai_embedding_model
            )
            embedding = response.data[0].embedding
            
            # Track API call statistics
            self.api_call_stats["openai_embedding_calls"] += 1
            self.api_call_stats["total_openai_calls"] += 1
            if hasattr(response, 'usage') and response.usage:
                tokens_used = getattr(response.usage, 'total_tokens', 0)
                self.api_call_stats["embedding_tokens_used"] += tokens_used
                self.api_call_stats["total_tokens_used"] += tokens_used
            
            # Cache the result in persistent cache
            self.cache_manager.set_embedding(text_hash, embedding)
            
            # Update performance metrics
            self.performance_metrics.embedding_time += time.time() - start_time
            
            self.logger.logger.debug(f"Generated embedding via API call (total calls: {self.api_call_stats['openai_embedding_calls']})")
            
            return embedding
        except Exception as e:
            self.logger.log_error(e, "generate_embedding")
            return None

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call"""
        if not texts:
            return []
            
        try:
            start_time = time.time()
            response = self.ai_instance.embeddings.create(
                input=texts,
                model=self.ai_embedding_model
            )
            embeddings = [data.embedding for data in response.data]
            
            # Track API call statistics for batch
            self.api_call_stats["openai_embedding_calls"] += 1
            self.api_call_stats["total_openai_calls"] += 1
            if hasattr(response, 'usage') and response.usage:
                tokens_used = getattr(response.usage, 'total_tokens', 0)
                self.api_call_stats["embedding_tokens_used"] += tokens_used
                self.api_call_stats["total_tokens_used"] += tokens_used
            
            # Cache the results
            for text, embedding in zip(texts, embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.cache_manager.set_embedding(text_hash, embedding)
            
            # Update performance metrics
            self.performance_metrics.embedding_time += time.time() - start_time
            
            self.logger.logger.debug(f"Generated {len(embeddings)} embeddings via batch API call")
            
            return embeddings
        except Exception as e:
            self.logger.log_error(e, "generate_embeddings_batch")
            return []

    async def generate_embedding_async(self, text: str) -> Optional[List[float]]:
        """Async version of embedding generation"""
        if not self.use_async:
            return self.generate_embedding(text)
            
        # For now, using sync version wrapped in async
        # In production, you'd want to use aiohttp for true async HTTP calls
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embedding, text)

    def _manage_context_window(self, context: str, max_tokens: int = 3000) -> str:
        """Truncate context if it exceeds token limits"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(context)
            
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                return encoding.decode(truncated_tokens)
            return context
        except Exception as e:
            self.logger.log_error(e, "_manage_context_window")
            return context[:max_tokens * 4]  # Fallback: rough character estimate

    def _optimize_chat_history(self):
        """Keep only the most relevant chat history"""
        if len(self.chat_history) > self.max_history_length:
            # Keep first few and last few messages
            self.chat_history = (
                self.chat_history[:4] + 
                self.chat_history[-16:]
            )

    def _extract_source_info(self, doc: Dict[str, Any], metadata: Dict[str, Any], doc_index: int) -> Dict[str, Any]:
        """Extract source information from document for citation purposes"""
        try:
            # Get basic file information
            filename = metadata.get("filename", "Unknown Document")
            file_path = doc.get("file_path", "")
            chunk_index = doc.get("chunk_index", 0)
            
            # Extract file extension to determine document type
            file_extension = os.path.splitext(filename)[1].lower() if filename != "Unknown Document" else ""
            
            # Determine page number for PDFs (estimate based on chunk index)
            page_number = None
            if file_extension == ".pdf":
                # Rough estimate: assuming ~3-4 chunks per page
                page_number = (chunk_index // 3) + 1
            
            # Create a readable source reference
            source_title = filename
            if page_number:
                source_title += f" (Page {page_number})"
            elif chunk_index > 0:
                source_title += f" (Section {chunk_index + 1})"
            
            # Generate file URL if possible (for local files, create file:// URL)
            file_url = None
            if file_path and os.path.exists(file_path):
                # Convert to file:// URL for local files
                file_url = f"file:///{file_path.replace(os.sep, '/')}"
            
            # Extract a preview of the content (first 150 characters)
            content_preview = doc.get("content", "")[:150]
            if len(doc.get("content", "")) > 150:
                content_preview += "..."
            
            # Calculate relevance score if available (from similarity search)
            relevance_score = doc.get("similarity", 0.0)
            
            return {
                "source_id": f"source_{doc_index + 1}",
                "filename": filename,
                "title": source_title,
                "file_path": file_path,
                "file_url": file_url,
                "file_type": file_extension.replace(".", "").upper() if file_extension else "UNKNOWN",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "content_preview": content_preview,
                "relevance_score": round(relevance_score, 3),
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.log_error(e, f"_extract_source_info for doc {doc_index}")
            return {
                "source_id": f"source_{doc_index + 1}",
                "filename": "Unknown Document",
                "title": "Unknown Source",
                "file_path": "",
                "file_url": None,
                "file_type": "UNKNOWN",
                "page_number": None,
                "chunk_index": 0,
                "content_preview": "",
                "relevance_score": 0.0,
                "metadata": {}
            }

    def format_sources_for_display(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for display in a user-friendly way"""
        if not sources:
            return "No sources available."
        
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            source_text = f"{i}. **{source['title']}**"
            
            if source['file_type'] != 'UNKNOWN':
                source_text += f" ({source['file_type']})"
            
            if source['page_number']:
                source_text += f" - Page {source['page_number']}"
            
            if source['relevance_score'] > 0:
                source_text += f" (Relevance: {source['relevance_score']:.1%})"
            
            if source['file_url']:
                source_text += f"\n   📎 [Open Document]({source['file_url']})"
            elif source['file_path']:
                source_text += f"\n   📁 Path: {source['file_path']}"
            
            if source['content_preview']:
                source_text += f"\n   📝 Preview: {source['content_preview']}"
            
            formatted_sources.append(source_text)
        
        return "\n\n".join(formatted_sources)

    @measure_performance
    def answer_this(self, question: str) -> Dict[str, Any]:
        """
        Enhanced answer generation with comprehensive optimizations and response caching
        """
        overall_start_time = time.time()
        
        # Sanitize input
        question = self.sanitize_input(question)
        if not question:
            return {"response": "Invalid input provided.", "error": "Input validation failed"}
        
        # Check response cache first
        question_hash = hashlib.md5(question.encode()).hexdigest()
        cached_response = self.cache_manager.get_response(question_hash)
        if cached_response is not None:
            self.cache_stats["response_hits"] += 1
            self.logger.logger.debug(f"Response cache HIT for question hash: {question_hash[:8]}...")
            cached_data = cached_response["response"]
            cached_data["cached"] = True
            cached_data["cache_timestamp"] = cached_response["timestamp"]
            return cached_data
        
        # Cache miss - continue with full processing
        self.cache_stats["response_misses"] += 1
        self.logger.logger.debug(f"Response cache MISS for question hash: {question_hash[:8]}...")
        
        try:
            # Step 1: Convert query to embedding
            embedding_start = time.time()
            query_embedding = self.generate_embedding(question)
            if not query_embedding:
                return {"response": "Failed to process your question. Please try again.", "error": "Embedding generation failed"}
            
            # Step 2: Vector search in Supabase
            search_start = time.time()
            response = self.supabase.rpc(
                "match_documents", 
                {
                    "query_embedding": query_embedding,
                    "match_threshold": self.vector_search_threshold,
                    "match_count": self.vector_search_match_count
                }
            ).execute()
            
            documents = response.data if response.data else []
            self.performance_metrics.search_time += time.time() - search_start
            self.performance_metrics.num_documents_retrieved = len(documents)
            
            if not documents:
                return {"response": self.ai_default_no_response, "documents_found": 0, "sources": []}
            
            # Step 3: Extract and manage context with source tracking
            context_parts = []
            source_documents = []
            
            for i, doc in enumerate(documents):
                context_parts.append(doc["content"])
                
                # Extract source information
                metadata = json.loads(doc.get("metadata", "{}")) if isinstance(doc.get("metadata"), str) else doc.get("metadata", {})
                source_info = self._extract_source_info(doc, metadata, i)
                source_documents.append(source_info)
            
            context = "\n".join(context_parts)
            context = self._manage_context_window(context)
            
            # Step 4: Optimize chat history
            self._optimize_chat_history()
            
            # Step 5: Construct message history
            messages = []
            
            if not self.is_initial_session:
                messages = [{"role": "system", "content": self.ai_system_role_prompt}]
                self.is_initial_session = True
                
            if self.chat_history:
                messages.append({"role": "system", "content": "Chat history follows:"})
                messages += self.chat_history
                
            # Add the current user query with retrieved context
            combined_input = f"[Reference Documents]\n{context}\n\n{question}"
            messages.append({"role": "user", "content": combined_input})
            
            # Step 6: Call OpenAI with error handling
            generation_start = time.time()
            chat_response = self.ai_instance.chat.completions.create(
                model=self.ai_main_model,
                messages=messages,
                max_tokens=self.ai_max_token_response,
                temperature=0.7
            )
            
            # Track API call statistics for chat
            self.api_call_stats["openai_chat_calls"] += 1
            self.api_call_stats["total_openai_calls"] += 1
            if hasattr(chat_response, 'usage') and chat_response.usage:
                tokens_used = getattr(chat_response.usage, 'total_tokens', 0)
                self.api_call_stats["chat_tokens_used"] += tokens_used
                self.api_call_stats["total_tokens_used"] += tokens_used
            
            self.performance_metrics.generation_time += time.time() - generation_start
            
            self.logger.logger.debug(f"Generated chat response via API call (total chat calls: {self.api_call_stats['openai_chat_calls']})")
            
            # Step 7: Get AI response and update history
            answer = chat_response.choices[0].message.content
            
            # Update chat history for memory
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # Calculate total time and log performance
            total_time = time.time() - overall_start_time
            self.performance_metrics.total_time = total_time
            self.performance_metrics.query_length = len(question)
            
            self.logger.log_query(question, total_time, len(documents))
            self.logger.log_performance(self.performance_metrics)
            
            # Prepare response data
            response_data = {
                "response": answer, 
                "documents_found": len(documents),
                "response_time": total_time,
                "performance_metrics": self.performance_metrics,
                "sources": source_documents,
                "cached": False
            }
            
            # Cache the response for future use
            self.cache_manager.set_response(question_hash, response_data)
            
            return response_data
            
        except Exception as e:
            self.logger.log_error(e, "answer_this")
            return {
                "response": "An error occurred while processing your question. Please try again.", 
                "error": str(e),
                "sources": []
            }

    @measure_performance
    def initialize_files(self, file_directory: str):
        """Process all files in the directory with improved error handling"""
        if not os.path.exists(file_directory):
            raise FileNotFoundError(f"Directory not found: {file_directory}")
            
        processed_files = self.load_processed_files()
        
        files_to_process = []
        for filename in os.listdir(file_directory):
            file_path = os.path.join(file_directory, filename)
            
            # Skip directories and already processed files
            if os.path.isdir(file_path) or filename in processed_files:
                continue
            files_to_process.append(filename)
        
        if not files_to_process:
            self.logger.logger.info("No new files to process")
            return
        
        self.logger.logger.info(f"Processing {len(files_to_process)} files")
        
        # Process files with progress tracking
        for filename in tqdm(files_to_process, desc="Processing files"):
            try:
                self.logger.logger.info(f"Processing file: {filename}")
                self.process_file(file_directory, filename)
                self.save_processed_file(filename)
            except Exception as e:
                self.logger.log_error(e, f"processing file {filename}")

    def load_processed_files(self) -> set:
        """Load the list of already processed files"""
        try:
            if os.path.exists(self.processed_files_list):
                with open(self.processed_files_list, "r", encoding="utf-8") as file:
                    return set(file.read().splitlines())
        except Exception as e:
            self.logger.log_error(e, "load_processed_files")
        return set()

    @measure_performance
    def process_file(self, directory: str, filename: str):
        """Process documents using LangChain with enhanced error handling"""
        try:
            filename = filename.strip()
            file_path = os.path.normpath(os.path.join(directory, filename))
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Use LangChain loaders based on file extension
            if file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load the document
            documents = loader.load()
            
            # Add enhanced metadata to each document
            for i, doc in enumerate(documents):
                doc.metadata["filename"] = filename
                doc.metadata["file_path"] = file_path
                doc.metadata["file_type"] = file_extension
                doc.metadata["processed_at"] = datetime.now().isoformat()
                doc.metadata["file_size"] = os.path.getsize(file_path)
                
                # For PDFs, try to extract page information
                if file_extension == ".pdf" and hasattr(doc.metadata, 'page') and doc.metadata.get('page') is not None:
                    doc.metadata["page_number"] = doc.metadata.get('page', 0) + 1  # LangChain uses 0-based indexing
                elif file_extension == ".pdf":
                    # Estimate page number based on document index (fallback)
                    doc.metadata["page_number"] = i + 1
                
                # Add document source URL (file path for local files)
                doc.metadata["source_url"] = f"file:///{file_path.replace(os.sep, '/')}"
            
            # Use RecursiveCharacterTextSplitter for intelligent chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_tokens,
                chunk_overlap=self.overlap_chunk_tokens,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            split_docs = splitter.split_documents(documents)
            
            self.logger.logger.info(f"Processing '{filename}' with {len(split_docs)} chunks")
            
            # Batch process embeddings for better performance
            texts = [doc.page_content for doc in split_docs]
            embeddings = self.generate_embeddings_batch(texts)
            
            # Upload to Supabase with enhanced metadata
            for i, (doc, embedding) in enumerate(zip(split_docs, embeddings)):
                if not embedding:
                    continue
                
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(split_docs)
                
                # Calculate estimated page for chunk (for PDFs)
                if file_extension == ".pdf":
                    # If we don't have page info from metadata, estimate based on chunk position
                    if "page_number" not in metadata:
                        # Estimate: roughly 3-4 chunks per page
                        estimated_page = (i // 3) + 1
                        metadata["estimated_page"] = estimated_page
                
                # Add content preview for search results
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                metadata["content_preview"] = content_preview
                
                data = {
                    "content": doc.page_content,
                    "embedding": embedding,
                    "metadata": json.dumps(metadata),
                    "file_path": file_path,
                    "chunk_index": i,
                    "filename": filename,
                    "file_type": file_extension
                }
                
                try:
                    self.supabase.table("document_embeddings").insert(data).execute()
                except Exception as e:
                    self.logger.log_error(e, f"uploading chunk {i} of {filename}")
                    
        except Exception as e:
            self.logger.log_error(e, f"process_file: {filename}")
            raise

    def save_processed_file(self, filename: str):
        """Add a file to the list of processed files"""
        try:
            with open(self.processed_files_list, "a", encoding="utf-8") as file:
                file.write(filename + "\n")
        except Exception as e:
            self.logger.log_error(e, "save_processed_file")

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller parts for processing"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            chunks = []
            for i in range(0, len(tokens), self.max_chunk_tokens - self.overlap_chunk_tokens):
                chunk_tokens = tokens[i:i + self.max_chunk_tokens]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            return chunks
        except Exception as e:
            self.logger.log_error(e, "chunk_text")
            return [text]  # Fallback: return original text

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report including cache statistics and API call tracking"""
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Calculate cache hit rates
        total_embedding_requests = self.cache_stats["embedding_hits"] + self.cache_stats["embedding_misses"]
        total_response_requests = self.cache_stats["response_hits"] + self.cache_stats["response_misses"]
        
        embedding_hit_rate = (self.cache_stats["embedding_hits"] / max(1, total_embedding_requests)) * 100
        response_hit_rate = (self.cache_stats["response_hits"] / max(1, total_response_requests)) * 100
        
        # Calculate API efficiency
        api_calls_saved_embedding = self.cache_stats["embedding_hits"]
        api_calls_saved_response = self.cache_stats["response_hits"]
        total_api_calls_saved = api_calls_saved_embedding + api_calls_saved_response
        
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "chat_history_length": len(self.chat_history),
            "latest_metrics": self.performance_metrics,
            "memory_usage": self._get_memory_usage(),
            "cache_performance": {
                "embedding_cache": {
                    "hits": self.cache_stats["embedding_hits"],
                    "misses": self.cache_stats["embedding_misses"],
                    "total_requests": total_embedding_requests,
                    "hit_rate_percentage": round(embedding_hit_rate, 1),
                    "api_calls_saved": api_calls_saved_embedding
                },
                "response_cache": {
                    "hits": self.cache_stats["response_hits"],
                    "misses": self.cache_stats["response_misses"],
                    "total_requests": total_response_requests,
                    "hit_rate_percentage": round(response_hit_rate, 1),
                    "api_calls_saved": api_calls_saved_response
                },
                "overall_cache_efficiency": {
                    "total_api_calls_saved": total_api_calls_saved,
                    "cache_hit_rate": round((self.cache_stats["embedding_hits"] + self.cache_stats["response_hits"]) / max(1, total_embedding_requests + total_response_requests) * 100, 1)
                }
            },
            "api_call_statistics": {
                **self.api_call_stats,
                "cost_efficiency": {
                    "potential_embedding_calls_without_cache": total_embedding_requests,
                    "actual_embedding_calls_made": self.api_call_stats["openai_embedding_calls"],
                    "potential_response_calls_without_cache": total_response_requests,
                    "actual_response_calls_made": self.api_call_stats["openai_chat_calls"],
                    "embedding_api_savings_percentage": round((api_calls_saved_embedding / max(1, total_embedding_requests)) * 100, 1),
                    "response_api_savings_percentage": round((api_calls_saved_response / max(1, total_response_requests)) * 100, 1)
                }
            },
            "cache_storage": cache_stats,
            "cache_statistics": {
                **self.cache_stats,
                **cache_stats,
                "total_requests": total_embedding_requests + total_response_requests,
                "embedding_hit_rate": f"{embedding_hit_rate:.1f}%",
                "response_hit_rate": f"{response_hit_rate:.1f}%"
            }
        }

    def _get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        import sys
        return {
            "embedding_cache_size_bytes": sys.getsizeof(self.embedding_cache),
            "chat_history_size_bytes": sys.getsizeof(self.chat_history)
        }

    def clear_cache(self):
        """Clear all caches (memory and persistent) and reset statistics"""
        self.cache_manager.clear_all_caches()
        self.embedding_cache = self.cache_manager.embedding_cache  # Update reference
        self.cache_stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "response_hits": 0,
            "response_misses": 0
        }
        # Note: Not clearing API call stats as they represent total usage across the session
        self.logger.logger.info("All caches cleared and cache statistics reset")
    
    def save_cache(self) -> bool:
        """Save cache to disk"""
        success = self.cache_manager.save_to_disk()
        if success:
            self.logger.logger.info("Cache saved successfully")
        else:
            self.logger.logger.error("Failed to save cache")
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return self.get_performance_report()["cache_statistics"]
    
    def get_api_call_summary(self) -> Dict[str, Any]:
        """Get a summary of API call statistics"""
        total_embedding_requests = self.cache_stats["embedding_hits"] + self.cache_stats["embedding_misses"]
        total_response_requests = self.cache_stats["response_hits"] + self.cache_stats["response_misses"]
        
        return {
            "session_summary": {
                "total_openai_api_calls": self.api_call_stats["total_openai_calls"],
                "embedding_api_calls": self.api_call_stats["openai_embedding_calls"],
                "chat_api_calls": self.api_call_stats["openai_chat_calls"],
                "total_tokens_used": self.api_call_stats["total_tokens_used"],
                "embedding_tokens_used": self.api_call_stats["embedding_tokens_used"],
                "chat_tokens_used": self.api_call_stats["chat_tokens_used"]
            },
            "efficiency_metrics": {
                "api_calls_saved_by_embedding_cache": self.cache_stats["embedding_hits"],
                "api_calls_saved_by_response_cache": self.cache_stats["response_hits"],
                "total_api_calls_saved": self.cache_stats["embedding_hits"] + self.cache_stats["response_hits"],
                "embedding_cache_efficiency": f"{(self.cache_stats['embedding_hits'] / max(1, total_embedding_requests)) * 100:.1f}%",
                "response_cache_efficiency": f"{(self.cache_stats['response_hits'] / max(1, total_response_requests)) * 100:.1f}%"
            }
        }
    
    def reset_api_call_stats(self):
        """Reset API call statistics (useful for testing or new session tracking)"""
        self.api_call_stats = {
            "openai_embedding_calls": 0,
            "openai_chat_calls": 0,
            "total_openai_calls": 0,
            "total_tokens_used": 0,
            "embedding_tokens_used": 0,
            "chat_tokens_used": 0
        }
        self.logger.logger.info("API call statistics reset")

    def reset_chat_history(self):
        """Reset chat history"""
        self.chat_history.clear()
        self.is_initial_session = False
        self.logger.logger.info("Chat history reset")

    async def cleanup(self):
        """Cleanup resources and save cache"""
        if self.session:
            await self.session.close()
        self.save_cache()  # Save cache before cleanup
        self.logger.logger.info("OptimizedRAGSystem cleanup completed")

    def __del__(self):
        """Destructor for cleanup - automatically save cache"""
        try:
            if hasattr(self, 'cache_manager'):
                self.save_cache()
        except:
            pass  # Ignore errors during destruction