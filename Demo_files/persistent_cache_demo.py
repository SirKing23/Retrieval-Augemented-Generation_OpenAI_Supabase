"""
Enhanced RAG System with Persistent Caching
This version saves the cache to disk and loads it on startup
"""

import os
import pickle
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

class PersistentCacheManager:
    """Manages persistent cache storage and retrieval"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self.cache = self._load_cache()
        
        # Setup logging
        self.logger = logging.getLogger('PersistentCache')
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                
                # Load metadata
                metadata = self._load_metadata()
                
                print(f"✅ Loaded cache from disk:")
                print(f"   📁 File: {self.cache_file}")
                print(f"   📊 Items: {len(cache)}")
                print(f"   📅 Last saved: {metadata.get('last_saved', 'Unknown')}")
                print(f"   💾 File size: {os.path.getsize(self.cache_file)} bytes")
                
                return cache
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}")
                return {}
        else:
            print(f"🆕 No existing cache found. Starting fresh.")
            return {}
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        metadata = {
            "last_saved": datetime.now().isoformat(),
            "cache_size": len(self.cache),
            "file_size_bytes": os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> bool:
        """Set item in cache"""
        # Check cache size limits
        if len(self.cache) > 1000:  # Limit number of items
            self._cleanup_old_entries()
        
        self.cache[key] = value
        return True
    
    def _cleanup_old_entries(self):
        """Remove old cache entries to manage size"""
        # Simple strategy: remove 20% of entries
        items_to_remove = len(self.cache) // 5
        keys_to_remove = list(self.cache.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        self.logger.info(f"Cleaned up {items_to_remove} old cache entries")
    
    def save_to_disk(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            self._save_metadata()
            
            print(f"💾 Cache saved to disk:")
            print(f"   📁 File: {self.cache_file}")
            print(f"   📊 Items: {len(self.cache)}")
            print(f"   💾 Size: {os.path.getsize(self.cache_file)} bytes")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            return False
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        file_size = os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        
        return {
            "items_in_cache": len(self.cache),
            "cache_file_path": self.cache_file,
            "cache_file_size_bytes": file_size,
            "cache_file_size_mb": file_size / (1024 * 1024),
            "cache_exists_on_disk": os.path.exists(self.cache_file)
        }

# Example usage class that integrates with RAG system
class EnhancedRAGWithPersistentCache:
    """RAG system with persistent caching capabilities"""
    
    def __init__(self):
        # Initialize persistent cache manager
        self.cache_manager = PersistentCacheManager(
            cache_dir="rag_cache",
            max_cache_size_mb=50
        )
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"🚀 Enhanced RAG System with Persistent Cache initialized")
    
    def generate_embedding_with_persistent_cache(self, text: str) -> List[float]:
        """Generate embedding with persistent caching"""
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check persistent cache
        cached_embedding = self.cache_manager.get(text_hash)
        if cached_embedding is not None:
            self.cache_hits += 1
            print(f"⚡ PERSISTENT CACHE HIT for: '{text[:50]}...'")
            return cached_embedding
        
        # Cache miss - simulate API call
        self.cache_misses += 1
        print(f"💾 CACHE MISS - API Call for: '{text[:50]}...'")
        
        # Simulate expensive API call
        import time
        time.sleep(0.1)  # Simulate API delay
        fake_embedding = [0.1 * i for i in range(100)]
        
        # Store in persistent cache
        self.cache_manager.set(text_hash, fake_embedding)
        
        return fake_embedding
    
    def save_cache(self):
        """Save cache to disk"""
        return self.cache_manager.save_to_disk()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        cache_stats = self.cache_manager.get_stats()
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **cache_stats,
            "session_cache_hits": self.cache_hits,
            "session_cache_misses": self.cache_misses,
            "session_hit_rate": f"{hit_rate:.1f}%",
            "total_session_requests": total_requests
        }
    
    def __del__(self):
        """Automatically save cache when object is destroyed"""
        try:
            self.save_cache()
            print(f"🔒 Cache automatically saved on exit")
        except:
            pass

def demo_persistent_cache():
    """Demonstrate persistent cache functionality"""
    print(f"🧪 PERSISTENT CACHE DEMONSTRATION")
    print(f"=" * 50)
    
    # Create RAG system instance
    rag = EnhancedRAGWithPersistentCache()
    
    # Test queries
    test_queries = [
        "What is ProductCenter?",
        "How to install ProductCenter?",
        "ProductCenter system requirements?",
        "What is ProductCenter?",  # Repeat - should hit cache
        "How to install ProductCenter?",  # Repeat - should hit cache
    ]
    
    print(f"\n🔍 Running test queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        rag.generate_embedding_with_persistent_cache(query)
    
    # Show statistics
    stats = rag.get_cache_stats()
    print(f"\n📊 SESSION STATISTICS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save cache
    print(f"\n💾 Saving cache...")
    rag.save_cache()
    
    print(f"\n✅ Cache persisted! Next program run will load this cache.")
    print(f"📁 Cache location: {rag.cache_manager.cache_file}")
    
    return rag

if __name__ == "__main__":
    demo_persistent_cache()
