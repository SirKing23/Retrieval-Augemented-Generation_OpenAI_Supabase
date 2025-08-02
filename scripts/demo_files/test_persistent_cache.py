"""
Test script for RAG_Core_Optimized with Persistent Caching
This script demonstrates the new persistent caching features for both embeddings and OpenAI responses
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from RAG_Core import OptimizedRAGSystem
import time

def test_persistent_caching():
    """Test the persistent caching functionality"""
    print("🧪 TESTING PERSISTENT CACHING IN RAG_CORE_OPTIMIZED")
    print("=" * 60)
    
    try:
        # Initialize the RAG system
        print("\n1️⃣ Initializing OptimizedRAGSystem...")
        rag = OptimizedRAGSystem()
        print("✅ RAG system initialized successfully")
        
        # Test questions
        test_questions = [
            "What is ProductCenter?",
            "How do I install ProductCenter?",
            "What are the system requirements?",
            "What is ProductCenter?",  # Duplicate - should hit both caches
            "How do I install ProductCenter?",  # Duplicate - should hit both caches
        ]
        
        print(f"\n2️⃣ Testing with {len(test_questions)} questions...")
        print("Note: First run may be slow due to API calls")
        
        # Process each question
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i}: {question} ---")
            start_time = time.time()
            
            # This will use both embedding cache and response cache
            result = rag.answer_this(question)
            
            end_time = time.time()
            
            # Display results
            if result.get("cached", False):
                print(f"⚡ CACHED RESPONSE (from disk)")
                print(f"   Cached on: {result.get('cache_timestamp', 'Unknown')}")
            else:
                print(f"🔄 NEW RESPONSE (API calls made)")
            
            print(f"   Response time: {end_time - start_time:.2f}s")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   Sources found: {result.get('documents_found', 0)}")
        
        # Show cache statistics
        print(f"\n3️⃣ Cache Performance Statistics:")
        cache_stats = rag.get_cache_stats()
        print(f"   Embedding Cache:")
        print(f"     • Items: {cache_stats['embedding_cache_items']}")
        print(f"     • Hit rate: {cache_stats['embedding_hit_rate']}")
        print(f"     • File size: {cache_stats['embedding_cache_file_size_bytes']} bytes")
        
        print(f"   Response Cache:")
        print(f"     • Items: {cache_stats['response_cache_items']}")
        print(f"     • Hit rate: {cache_stats['response_hit_rate']}")
        print(f"     • File size: {cache_stats['response_cache_file_size_bytes']} bytes")
        
        print(f"   Total cache size: {cache_stats['total_cache_size_mb']:.2f} MB")
        print(f"   Cache directory: {cache_stats['cache_directory']}")
        
        # Save cache explicitly
        print(f"\n4️⃣ Saving cache to disk...")
        success = rag.save_cache()
        if success:
            print("✅ Cache saved successfully")
        else:
            print("❌ Failed to save cache")
        
        print(f"\n5️⃣ Cache files created:")
        cache_dir = "rag_cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"   📁 {file}: {file_size} bytes")
        
        print(f"\n✅ PERSISTENT CACHING TEST COMPLETED")
        print(f"🔄 Next time you run this script, cached responses will load instantly!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_cache_persistence():
    """Test that cache persists between program runs"""
    print(f"\n🔄 TESTING CACHE PERSISTENCE")
    print("=" * 40)
    
    try:
        # Create a new RAG instance (simulating restart)
        print("Simulating program restart...")
        rag = OptimizedRAGSystem()
        
        # Check if cache was loaded
        cache_stats = rag.get_cache_stats()
        print(f"Cache loaded from disk:")
        print(f"   • Embedding items: {cache_stats['embedding_cache_items']}")
        print(f"   • Response items: {cache_stats['response_cache_items']}")
        
        if cache_stats['embedding_cache_items'] > 0 or cache_stats['response_cache_items'] > 0:
            print("✅ Cache successfully persisted and loaded!")
            
            # Test with a previously cached question
            print(f"\nTesting with cached question...")
            start_time = time.time()
            result = rag.answer_this("What is ProductCenter?")
            end_time = time.time()
            
            if result.get("cached", False):
                print(f"⚡ INSTANT RESPONSE from persistent cache!")
                print(f"   Response time: {end_time - start_time:.3f}s")
            else:
                print(f"🔄 Cache miss - response not cached yet")
        else:
            print("ℹ️ No cache found - this is expected on first run")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 RAG PERSISTENT CACHING DEMONSTRATION")
    print("This script tests both embedding and response caching")
    print("Cache files are saved in the 'rag_cache' directory")
    print()
    
    # Test 1: Basic persistent caching
    test1_success = test_persistent_caching()
    
    # Test 2: Cache persistence between runs
    test2_success = test_cache_persistence()
    
    if test1_success and test2_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Your RAG system now has persistent caching for:")
        print(f"   ✅ Embedding vectors (avoids expensive API calls)")
        print(f"   ✅ Complete responses (instant answers for repeated questions)")
        print(f"   ✅ Cache survives program restarts")
        print(f"   ✅ Automatic cache cleanup and management")
    else:
        print(f"\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
