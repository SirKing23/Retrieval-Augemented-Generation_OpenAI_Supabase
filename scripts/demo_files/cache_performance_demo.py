"""
Cache Performance Demonstration for RAG System
This script shows the dramatic performance improvement from caching
"""

import time
import hashlib
from typing import List, Dict
from unittest.mock import Mock, patch

# Simulate the caching behavior without actual API calls
class CacheDemo:
    def __init__(self):
        self.embedding_cache = {}
        self.api_call_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def simulate_expensive_api_call(self, text: str) -> List[float]:
        """Simulate OpenAI API call with artificial delay"""
        self.api_call_count += 1
        
        # Simulate network latency and processing time
        time.sleep(0.5)  # Simulates 500ms API call
        
        # Return fake embedding (normally 1536 dimensions)
        fake_embedding = [0.1 * i for i in range(100)]  # Simplified for demo
        return fake_embedding
    
    def generate_embedding_without_cache(self, text: str) -> List[float]:
        """Generate embedding WITHOUT caching - always calls API"""
        print(f"🔄 API Call for: '{text[:50]}...'")
        return self.simulate_expensive_api_call(text)
    
    def generate_embedding_with_cache(self, text: str) -> List[float]:
        """Generate embedding WITH caching - cache first, API if needed"""
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if text_hash in self.embedding_cache:
            self.cache_hits += 1
            print(f"⚡ CACHE HIT for: '{text[:50]}...'")
            return self.embedding_cache[text_hash]
        
        # Cache miss - call API
        self.cache_misses += 1
        print(f"💾 CACHE MISS - API Call for: '{text[:50]}...'")
        embedding = self.simulate_expensive_api_call(text)
        
        # Store in cache
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "api_calls": self.api_call_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self.embedding_cache),
            "api_calls_saved": self.cache_hits
        }

def demo_without_cache():
    """Demonstrate performance WITHOUT caching"""
    print("🐌 DEMO: WITHOUT CACHING")
    print("=" * 50)
    
    demo = CacheDemo()
    test_questions = [
        "What is ProductCenter?",
        "How to install ProductCenter?", 
        "What is ProductCenter?",  # Duplicate - will still call API
        "Tell me about ProductCenter",  # Similar but different
        "What is ProductCenter?",  # Another duplicate - still calls API
    ]
    
    start_time = time.time()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nRequest {i}:")
        demo.generate_embedding_without_cache(question)
    
    total_time = time.time() - start_time
    stats = demo.get_stats()
    
    print(f"\n📊 Results WITHOUT Caching:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   API calls made: {stats['api_calls']}")
    print(f"   Average time per request: {total_time/len(test_questions):.2f}s")
    print(f"   Cost: ${stats['api_calls'] * 0.0001:.6f} (simulated)")
    
    return total_time, stats

def demo_with_cache():
    """Demonstrate performance WITH caching"""
    print("\n\n🚀 DEMO: WITH CACHING")
    print("=" * 50)
    
    demo = CacheDemo()
    test_questions = [
        "What is ProductCenter?",
        "How to install ProductCenter?",
        "What is ProductCenter?",  # Duplicate - CACHE HIT!
        "Tell me about ProductCenter",  # Different text
        "What is ProductCenter?",  # Another duplicate - CACHE HIT!
    ]
    
    start_time = time.time()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nRequest {i}:")
        demo.generate_embedding_with_cache(question)
    
    total_time = time.time() - start_time
    stats = demo.get_stats()
    
    print(f"\n📊 Results WITH Caching:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   API calls made: {stats['api_calls']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']}")
    print(f"   Average time per request: {total_time/len(test_questions):.2f}s")
    print(f"   Cost: ${stats['api_calls'] * 0.0001:.6f} (simulated)")
    print(f"   API calls saved: {stats['api_calls_saved']}")
    
    return total_time, stats

def demo_real_world_scenario():
    """Demonstrate real-world caching benefits"""
    print("\n\n🌍 DEMO: REAL-WORLD SCENARIO")
    print("=" * 50)
    print("Simulating a typical user session with repeated queries")
    
    demo = CacheDemo()
    
    # Simulate realistic user behavior - users often ask similar questions
    session_questions = [
        "What is ProductCenter?",
        "How do I install ProductCenter?",
        "What are ProductCenter system requirements?",
        "What is ProductCenter?",  # User asks again
        "How to configure ProductCenter database?",
        "What is ProductCenter used for?",  # Similar to first
        "How do I install ProductCenter?",  # Exact repeat
        "ProductCenter installation guide",  # Slightly different
        "What is ProductCenter?",  # User really wants to know this
        "How do I install ProductCenter?",  # Another repeat
    ]
    
    start_time = time.time()
    
    for i, question in enumerate(session_questions, 1):
        print(f"\nUser query {i}: '{question}'")
        demo.generate_embedding_with_cache(question)
    
    total_time = time.time() - start_time
    stats = demo.get_stats()
    
    print(f"\n🎯 Real-World Session Results:")
    print(f"   Session duration: {total_time:.2f} seconds")
    print(f"   Total queries: {len(session_questions)}")
    print(f"   Unique queries (API calls): {stats['api_calls']}")
    print(f"   Cached responses: {stats['cache_hits']}")
    print(f"   Cache efficiency: {stats['cache_hit_rate']}")
    print(f"   Time saved: {stats['cache_hits'] * 0.5:.1f} seconds")
    print(f"   Money saved: ${stats['cache_hits'] * 0.0001:.6f}")
    
    return stats

def main():
    """Run all cache demonstrations"""
    print("🧪 CACHE PERFORMANCE DEMONSTRATION")
    print("This demo shows why caching is crucial for RAG systems\n")
    
    # Demo 1: Without cache
    time_without, stats_without = demo_without_cache()
    
    # Demo 2: With cache  
    time_with, stats_with = demo_with_cache()
    
    # Demo 3: Real-world scenario
    real_world_stats = demo_real_world_scenario()
    
    # Summary comparison
    print("\n\n📈 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    improvement = ((time_without - time_with) / time_without) * 100
    print(f"⚡ Speed improvement: {improvement:.1f}% faster with caching")
    print(f"💰 Cost reduction: {stats_with['cache_hits']} fewer API calls")
    print(f"🎯 Cache hit rate: {real_world_stats['cache_hit_rate']}")
    
    print(f"\n🔑 Key Insights:")
    print(f"   • Without cache: Every request = API call = time + money")
    print(f"   • With cache: Repeat requests = instant + free")
    print(f"   • Real users often ask similar questions")
    print(f"   • Cache hit rates of 40-70% are typical")
    print(f"   • Each cache hit saves ~500ms and $0.0001")
    
    print(f"\n💡 Why This Matters:")
    print(f"   • Faster responses = better user experience")
    print(f"   • Lower costs = more sustainable application")  
    print(f"   • Reduced API load = more reliable system")
    print(f"   • Scalability = handle more users with same resources")

if __name__ == "__main__":
    main()
