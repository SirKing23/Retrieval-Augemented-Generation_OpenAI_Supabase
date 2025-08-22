#!/usr/bin/env python3
"""
Test script for the improved RAG system
This script tests the enhanced ChromaDB querying and similarity search functionality
"""

import os
import sys
from src.rag_core.RAG_Core import RAGSystem, RAGSystemConfig

def test_rag_improvements():
    """Test the improved RAG system functionality"""
    
    print("🔧 Testing Improved RAG System")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        config = RAGSystemConfig()
        # Lower threshold for testing
        config.vector_search_threshold = 0.3
        config.vector_search_match_count = 10
        
        rag = RAGSystem(config=config)
        
        print(f"✅ RAG System initialized successfully")
        print(f"📊 Search threshold: {rag.vector_search_threshold}")
        print(f"📊 Match count: {rag.vector_search_match_count}")
        
        # Get diagnostics
        print("\n🔍 ChromaDB Diagnostics:")
        diagnostics = rag.get_chromadb_diagnostics()
        print(f"Collection status: {diagnostics.get('collection_status', 'unknown')}")
        print(f"Total documents: {diagnostics.get('total_documents', 0)}")
        print(f"Unique files: {diagnostics.get('unique_files_in_sample', 0)}")
        print(f"Files in collection: {diagnostics.get('file_names_in_sample', [])}")
        
        if diagnostics.get('total_documents', 0) == 0:
            print("\n⚠️  Collection is empty! Processing documents...")
            
            # Check if there are documents to process
            documents_dir = os.getenv("DOCUMENTS_DIR", "./data/Knowledge_Base_Files")
            if os.path.exists(documents_dir):
                rag.initialize_files(documents_dir)
                print("✅ Documents processed")
                
                # Get updated diagnostics
                diagnostics = rag.get_chromadb_diagnostics()
                print(f"Updated total documents: {diagnostics.get('total_documents', 0)}")
            else:
                print(f"❌ Documents directory not found: {documents_dir}")
                return
        
        # Test queries with different complexity
        test_queries = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "Tell me about deep learning",
            "What are neural networks?",
            "Explain computer vision",
            "Simple question",  # Very short query
            "a",  # Too short query
        ]
        
        print(f"\n🧪 Testing {len(test_queries)} queries:")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # First test similarity search
            search_results = rag.search_similar_documents(query, top_k=5)
            
            if "error" in search_results:
                print(f"   ❌ Search error: {search_results['error']}")
                continue
                
            print(f"   📊 Found {search_results.get('total_results', 0)} results")
            print(f"   📊 Above threshold: {search_results.get('above_threshold', 0)}")
            
            if search_results.get('results'):
                best_result = search_results['results'][0]
                print(f"   🎯 Best match: {best_result['similarity']:.3f} similarity")
                print(f"   📄 From: {best_result['filename']}")
            
            # Now test full answer generation
            print("   🤖 Generating answer...")
            response = rag.answer_this(query)
            
            print(f"   📝 Documents found: {response.get('documents_found', 0)}")
            
            if response.get('cached'):
                print("   💾 (Cached response)")
            
            if response.get('debug_info'):
                debug = response['debug_info']
                print(f"   🔍 Debug - Best similarity: {debug.get('best_similarity', 0):.3f}")
                print(f"   🔍 Debug - Collection size: {debug.get('collection_size', 0)}")
            
            # Show first 100 chars of response
            response_text = response.get('response', 'No response')
            if len(response_text) > 100:
                response_text = response_text[:100] + "..."
            print(f"   💬 Response: {response_text}")
            
            print("   " + "-" * 30)
        
        # Test performance report
        print(f"\n📈 Performance Report:")
        perf_report = rag.get_performance_report()
        cache_perf = perf_report.get('cache_performance', {})
        
        if 'embedding_cache' in cache_perf:
            emb_cache = cache_perf['embedding_cache']
            print(f"Embedding cache hit rate: {emb_cache.get('hit_rate_percentage', 0)}%")
        
        if 'response_cache' in cache_perf:
            resp_cache = cache_perf['response_cache']
            print(f"Response cache hit rate: {resp_cache.get('hit_rate_percentage', 0)}%")
        
        # Save cache
        if rag.save_cache():
            print("✅ Cache saved successfully")
        
        print(f"\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_improvements()
