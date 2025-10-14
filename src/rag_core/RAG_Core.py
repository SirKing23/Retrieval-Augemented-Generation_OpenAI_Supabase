from openai import OpenAI
import os

import json
from tqdm import tqdm
import tiktoken

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


# Load environment variables
load_dotenv()

@dataclass
class PerformanceMetrics:
    """Data class to track performance metrics"""
  
    search_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    num_documents_retrieved: int = 0
    query_length: int = 0

class RAGSystemConfig:
    """Configuration management class"""
    def __init__(self):
        self.ai_api_key = os.getenv("OPENAI_API_KEY")       
        self.main_model = os.getenv("MAIN_MODEL", "gpt-4o-mini")      
        self.max_token_response = int(os.getenv("MAX_TOKEN_RESPONSE", "700"))      
        
     
    def validate_config(self):
        """Validate all configuration parameters"""
        required_vars = ["OPENAI_API_KEY"]
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
            f"Search: {metrics.search_time:.2f}s, "
            f"Generation: {metrics.generation_time:.2f}s, "
            f"Total: {metrics.total_time:.2f}s"
        )



class PersistentCacheManager:
    """Manages persistent cache storage for responses"""
    
    def __init__(self, cache_dir: str = None, max_cache_size_mb: int = 100):
        # Get cache directory from environment variable or use default
        if cache_dir is None:
            cache_dir = os.getenv("CACHE_DIR", "./cache")
        
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Cache file paths
       
        self.response_cache_file = os.path.join(cache_dir, "response_cache.pkl")
        self.cache_metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup logging first
        self.logger = logging.getLogger('PersistentCache')
        
        # Load existing caches

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
            
            response_size = os.path.getsize(self.response_cache_file) if os.path.exists(self.response_cache_file) else 0
            
            metadata = {
                "last_saved": datetime.now().isoformat(),
                "response_cache_size": len(self.response_cache),            
                "response_file_size_bytes": response_size,               
            }            
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
       
    def get_response(self, question_hash: str) -> Optional[Dict[str, Any]]:
        """Get response from cache"""
        return self.response_cache.get(question_hash)
    
    def set_response(self, question_hash: str, response_data: Dict[str, Any]) -> bool:
        """Set response in cache and save to disk"""
        # Store response with timestamp
        cache_entry = {
            "response": response_data,
            "timestamp": datetime.now().isoformat(),
            "cached": True
        }
        self.response_cache[question_hash] = cache_entry
        
        # Save to disk immediately to ensure persistence
        try:
            self.save_to_disk()
            return True
        except Exception as e:
            self.logger.error(f"Failed to save response cache to disk: {e}")
            return False
    
    def _cleanup_old_entries(self):
        """Remove old cache entries to manage size"""
        # Simple strategy: remove 20% of oldest entries from each cache
        for cache in [self.response_cache]:
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
           
            response_success = self._save_cache(self.response_cache, self.response_cache_file)
            
            # Save metadata
            self._save_metadata()            
           
        except Exception as e:
            self.logger.error(f"Failed to save caches: {e}")
            return False
    
    def clear_all_caches(self):
        """Clear all caches"""
     
        self.response_cache.clear()        
        # Remove files
        for file_path in [self.response_cache_file, self.cache_metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
       
        response_size = os.path.getsize(self.response_cache_file) if os.path.exists(self.response_cache_file) else 0
        
        return {            
            "response_cache_items": len(self.response_cache),
            "response_cache_file_size_bytes": response_size,      
            "total_cache_size_mb": ( response_size) / (1024 * 1024),
            "cache_directory": self.cache_dir
        }

class RAGSystem:

    def __init__(self, config: Optional[RAGSystemConfig] = None, use_async: bool = False):
        """
        Initialize the RAG system
        
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
  
        self.ai_max_token_response = self.config.max_token_response
        self.ai_instance = OpenAI(api_key=self.config.ai_api_key)
        self.ai_default_no_response = "I couldn't find any relevant information in the knowledge base regarding your question."
        self.ai_system_role_prompt = "You are a helpful assistant with access to relevant documents and chat history. If the relevant documents and chat history has nothing to do with the user query, respond politely that the query is out of context."
        
        
        # Chat History parameters with improved management
        self.chat_history = []
        self.is_initial_session = False
        self.max_history_length = 20
        
        # Async session for async operations
        self.use_async = use_async
        self.session = None
        
        # Initialize persistent cache manager
        cache_dir = os.getenv("CACHE_DIR", "./cache/rag_cache")
        
        # If cache_dir is relative, make it relative to the project root
        if not os.path.isabs(cache_dir):
            # Find project root (directory containing .env file)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root and not os.path.exists(os.path.join(project_root, '.env')):
                parent = os.path.dirname(project_root)
                if parent == project_root:  # Reached filesystem root
                    break
                project_root = parent
            
            if os.path.exists(os.path.join(project_root, '.env')):
                cache_dir = os.path.join(project_root, cache_dir.lstrip('./'))
            else:
                # Fallback to absolute path
                cache_dir = os.path.abspath(cache_dir)
        
        self.cache_manager = PersistentCacheManager(cache_dir=cache_dir, max_cache_size_mb=100)
        
       
        
        # Statistics for cache performance and API calls
        self.cache_stats = {           
            "response_hits": 0,
            "response_misses": 0
        }
        
        # API call tracking
        self.api_call_stats = {            
            "openai_chat_calls": 0,
            "total_openai_calls": 0,
            "total_tokens_used": 0,           
            "chat_tokens_used": 0
        }
        
        # Query routing statistics
        self.routing_stats = {
            "total_queries": 0,                      
            "cache_hits": 0,
            "direct_route": 0            
        }
        
        self.logger.logger.info("RAG System initialized successfully with persistent caching")

    def _validate_initialization_parameters(self):
        """Validate initialization parameters"""
        if not self.config.ai_api_key or not isinstance(self.config.ai_api_key, str):
            raise ValueError("AI_API_Key must be a non-empty string")
      
        if self.config.max_token_response <= 0:
            raise ValueError("max_token_response must be positive")
       
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and basic security"""
        if not api_key or len(api_key) < 20:
            return False
       
        return True

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            return ""
        # Remove or escape potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        # Limit length to prevent excessive token usage
        return sanitized[:10000]  # Limit to 10k characters
  
    def _build_system_prompt(self) -> str:
        """
        Chain of Execution: Step 6
        Build the system prompt for the OpenAI model.
        
        Returns:
            str: The system prompt for SQL generation
        """
        return """You are an intelligent SQL query generator assistant.

            Your goal is to convert natural language questions into executable SQL statements.

            ### Your Responsibilities:
            1. Accept two inputs:
            - **User Query (Natural Language):** The user's question or request in plain English.
            - **Database Schema:** A structured description of the available database tables, their columns, and relationships.

            2. **Step 1 — Determine Intent:**
            - Analyze if the user's query is intended to retrieve, insert, update, or delete data.
            - If the query is not about SQL or database operations, respond with:
                "Not a database-related query."

            3. **Step 2 — Generate SQL:**
            - Convert the natural language query into a syntactically correct SQL query.
            - Use table and column names from the provided schema.
            - Avoid guessing column names that are not present in the schema.
            - Use descriptive aliases and proper JOINs when necessary.
            - If data aggregation or filtering is implied, infer the correct SQL functions (e.g. COUNT, SUM, GROUP BY, LIKE).

            4. **Step 3 — Validate Output:**
            - Ensure the SQL statement is valid and formatted clearly.
            - Never include natural language explanation in the final output.
            - Output only the SQL query.

            5. **Output Format:**
            - If the query is valid → output only the SQL query (formatted).
            - If not a valid SQL-related question → output: "Not a database-related query."

            IMPORTANT: Return ONLY the SQL query or "Not a database-related query." Do not include any explanations, comments, or additional text."""
                
    def database_schema(self) -> Dict[str, any]:
        """
        Returns database schema for SQL generation.
        The schema describes tables, columns, and relationships.
        """
        return {
            "tables": [
                {
                    "name": "employees",
                    "columns": [
                        {"name": "employee_id", "type": "INTEGER", "primary_key": True},
                        {"name": "first_name", "type": "VARCHAR"},
                        {"name": "last_name", "type": "VARCHAR"},
                        {"name": "department_id", "type": "INTEGER"},
                        {"name": "hire_date", "type": "DATE"},
                        {"name": "salary", "type": "DECIMAL"}
                    ]
                },
                {
                    "name": "departments",
                    "columns": [
                        {"name": "department_id", "type": "INTEGER", "primary_key": True},
                        {"name": "department_name", "type": "VARCHAR"},
                        {"name": "manager_id", "type": "INTEGER"}
                    ]
                },
                {
                    "name": "projects",
                    "columns": [
                        {"name": "project_id", "type": "INTEGER", "primary_key": True},
                        {"name": "project_name", "type": "VARCHAR"},
                        {"name": "start_date", "type": "DATE"},
                        {"name": "end_date", "type": "DATE"}
                    ]
                },
                {
                    "name": "employee_projects",
                    "columns": [
                        {"name": "employee_id", "type": "INTEGER"},
                        {"name": "project_id", "type": "INTEGER"},
                        {"name": "role", "type": "VARCHAR"}
                    ]
                }
            ],
            "relationships": [
                {
                    "from_table": "employees",
                    "from_column": "department_id",
                    "to_table": "departments",
                    "to_column": "department_id",
                    "type": "many-to-one"
                },
                {
                    "from_table": "departments",
                    "from_column": "manager_id",
                    "to_table": "employees",
                    "to_column": "employee_id",
                    "type": "one-to-one"
                },
                {
                    "from_table": "employee_projects",
                    "from_column": "employee_id",
                    "to_table": "employees",
                    "to_column": "employee_id",
                    "type": "many-to-one"
                },
                {
                    "from_table": "employee_projects",
                    "from_column": "project_id",
                    "to_table": "projects",
                    "to_column": "project_id",
                    "type": "many-to-one"
                }
            ]
        }

    def queryConverter(self, question: str) -> Dict[str, Any]:
        """
        Convert natural language query to sql
        """
        try:
            start_time = time.time()            
         
            system_prompt = self._build_system_prompt()
            
            # Build the user prompt including the database schema
            schema = self.database_schema()
            user_prompt = (
                f"User Query: {question}\n\n"
                f"Database Schema:\n{json.dumps(schema, indent=2)}"
            )

            messages = [{"role": "system", "content": system_prompt}]
            if self.chat_history:
                messages.append({"role": "system", "content": "Chat history follows:"})               
                messages += self.chat_history[-5:]
            messages.append({"role": "user", "content": user_prompt})
            
            # Generate response
            chat_response = self.ai_instance.chat.completions.create(
                model=self.ai_main_model,
                messages=messages,
                max_tokens=self.ai_max_token_response,
                temperature=0.7
            )
            
            # Track API call statistics
            self.api_call_stats["openai_chat_calls"] += 1
            self.api_call_stats["total_openai_calls"] += 1
            if hasattr(chat_response, 'usage') and chat_response.usage:
                tokens_used = getattr(chat_response.usage, 'total_tokens', 0)
                self.api_call_stats["chat_tokens_used"] += tokens_used
                self.api_call_stats["total_tokens_used"] += tokens_used
            
            answer = chat_response.choices[0].message.content
            response_time = time.time() - start_time
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            self.logger.logger.info(f"Direct answer generated in {response_time:.2f}s")
            
            return {
                "response": answer,
                "route_used": "direct",                
                "response_time": response_time,               
                "cached": False
            }
            
        except Exception as e:
            self.logger.log_error(e, "answer_direct")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "error": str(e),
                "route_used": "direct",              
            }

   
     
        return os.getenv("DOCUMENTS_DIR", "./data/documents")  
  
    def answer_this(self, question: str) -> Dict[str, Any]:
        """
        Enhanced answer generation with query routing and comprehensive optimizations.
        Routes queries to either direct LLM response or knowledge base retrieval based on query classification.
        """
        overall_start_time = time.time()
        
        # Track total queries
        self.routing_stats["total_queries"] += 1
        
        # Step 1: Sanitize user query
        question = self.sanitize_input(question)
        if not question:
            return {"response": "Invalid input provided.", "error": "Input validation failed"}
        
        # Step 2: Check persistent cache first
        question_hash = hashlib.md5(question.encode()).hexdigest()
        cached_response = self.cache_manager.get_response(question_hash)
        if cached_response is not None:
            self.cache_stats["response_hits"] += 1
            self.routing_stats["cache_hits"] += 1
            self.logger.logger.debug(f"Response cache HIT for question hash: {question_hash[:8]}...")
            cached_data = cached_response["response"]
            cached_data["cached"] = True
            cached_data["cache_timestamp"] = cached_response["timestamp"]
            return cached_data
        
        # Cache miss - continue with LLM for processing
        self.cache_stats["response_misses"] += 1     
        self.logger.logger.debug(f"Response cache MISS for question hash: {question_hash[:8]}...")       
        self.routing_stats["direct_route"] += 1

        # Step 3: Convert query to sql
        response_data = self.queryConverter(question)
        
        self.cache_manager.set_response(question_hash, response_data)
        return response_data  
       
    def clear_cache(self):
        """Clear all caches (memory and persistent) and reset statistics"""
        self.cache_manager.clear_all_caches()      
        self.cache_stats = {           
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
       
        total_response_requests = self.cache_stats["response_hits"] + self.cache_stats["response_misses"]
        
        return {
            "session_summary": {
                "total_openai_api_calls": self.api_call_stats["total_openai_calls"],                
                "chat_api_calls": self.api_call_stats["openai_chat_calls"],
                "total_tokens_used": self.api_call_stats["total_tokens_used"],
              
                "chat_tokens_used": self.api_call_stats["chat_tokens_used"]
            },
            "efficiency_metrics": {
              
                "api_calls_saved_by_response_cache": self.cache_stats["response_hits"],
                "total_api_calls_saved":  self.cache_stats["response_hits"],               
                "response_cache_efficiency": f"{(self.cache_stats['response_hits'] / max(1, total_response_requests)) * 100:.1f}%"
            }
        }
    
    def reset_api_call_stats(self):
        """Reset API call statistics (useful for testing or new session tracking)"""
        self.api_call_stats = {
            
            "openai_chat_calls": 0,
            "total_openai_calls": 0,
            "total_tokens_used": 0,           
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

    