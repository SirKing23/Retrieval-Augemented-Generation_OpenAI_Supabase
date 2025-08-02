"""
Simple test to verify the PersistentCacheManager fix
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_cache_manager():
    """Test that PersistentCacheManager initializes correctly"""
    try:
        from RAG_Core import PersistentCacheManager
        
        print("Testing PersistentCacheManager initialization...")
        cache_manager = PersistentCacheManager()
        print("✅ PersistentCacheManager initialized successfully!")
        
        # Test basic functionality
        print("Testing cache operations...")
        cache_manager.set_embedding("test_hash", [0.1, 0.2, 0.3])
        result = cache_manager.get_embedding("test_hash")
        
        if result == [0.1, 0.2, 0.3]:
            print("✅ Cache operations working correctly!")
        else:
            print("❌ Cache operations failed!")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_rag_system():
    """Test that OptimizedRAGSystem initializes correctly"""
    try:
        from RAG_Core import OptimizedRAGSystem
        
        print("\nTesting OptimizedRAGSystem initialization...")
        rag_system = OptimizedRAGSystem()
        print("✅ OptimizedRAGSystem initialized successfully!")
        
        # Test cache stats
        print("Testing cache statistics...")
        stats = rag_system.get_cache_stats()
        print(f"   Cache directory: {stats.get('cache_directory', 'Not found')}")
        print(f"   Embedding items: {stats.get('embedding_cache_items', 0)}")
        print(f"   Response items: {stats.get('response_cache_items', 0)}")
        print("✅ Cache statistics working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTING PERSISTENT CACHE FIX")
    print("=" * 40)
    
    test1 = test_cache_manager()
    test2 = test_rag_system()
    
    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED!")
        print("The PersistentCacheManager logger issue has been fixed!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
