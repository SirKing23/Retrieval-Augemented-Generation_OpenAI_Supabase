"""
Test script to verify source tracking functionality
Run this after setting up your environment variables
"""

import os
from RAG_Core import OptimizedRAGSystem, RAGSystemConfig

def test_source_tracking():
    """Test the new source tracking features"""
    print("🧪 Testing Source Tracking Features")
    print("=" * 50)
    
    try:
        # Initialize the system
        rag_system = OptimizedRAGSystem()
        print("✅ RAG System initialized successfully")
        
        # Test question
        question = "How to use GroupLoadByName? also give example of how to use it."
        print(f"\n🤔 Test Question: {question}")
        
        # Get response with sources
        response = rag_system.answer_this(question)
        
        # Check if sources are included
        sources = response.get('sources', [])
        print(f"\n📊 Response Analysis:")
        print(f"   Documents found: {response.get('documents_found', 0)}")
        print(f"   Sources provided: {len(sources)}")
        print(f"   Response time: {response.get('response_time', 0):.2f}s")
        
        if sources:
            print(f"\n✅ Source tracking working! Found {len(sources)} sources:")
            
            for i, source in enumerate(sources, 1):
                print(f"\n📄 Source {i}:")
                print(f"   📁 Filename: {source.get('filename', 'Unknown')}")
                print(f"   📂 File Type: {source.get('file_type', 'Unknown')}")
                print(f"   📄 Page: {source.get('page_number', 'N/A')}")
                print(f"   🎯 Relevance: {source.get('relevance_score', 0):.1%}")
                print(f"   🔗 URL: {source.get('file_url', 'No URL')}")
                print(f"   📝 Preview: {source.get('content_preview', 'No preview')[:100]}...")
                
                # Test source info extraction
                required_fields = ['source_id', 'filename', 'title', 'file_type']
                missing_fields = [field for field in required_fields if field not in source]
                
                if missing_fields:
                    print(f"   ⚠️  Missing fields: {missing_fields}")
                else:
                    print(f"   ✅ All required fields present")
            
            # Test formatted sources
            formatted = rag_system.format_sources_for_display(sources)
            print(f"\n📋 Formatted Sources Preview:")
            print(formatted[:300] + "..." if len(formatted) > 300 else formatted)
            
        else:
            print("❌ No sources found - this might indicate:")
            print("   1. No documents have been processed yet")
            print("   2. No relevant documents found for the query")
            print("   3. Database connection issues")
            
        # Test error handling with invalid input
        print(f"\n🧪 Testing error handling...")
        error_response = rag_system.answer_this("")
        if 'error' in error_response and 'sources' in error_response:
            print("✅ Error handling includes empty sources list")
        else:
            print("❌ Error handling missing sources field")
            
        print(f"\n🎉 Source tracking test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\n💡 Make sure you have:")
        print("   - Valid API keys in environment variables")
        print("   - Supabase database properly configured")
        print("   - Documents processed in your database")

def test_source_metadata():
    """Test the metadata extraction functionality"""
    print(f"\n🔬 Testing Source Metadata Extraction")
    print("=" * 50)
    
    try:
        rag_system = OptimizedRAGSystem()
        
        # Create sample document data for testing
        sample_doc = {
            "content": "This is a sample document content for testing purposes.",
            "file_path": "/test/sample.pdf",
            "chunk_index": 2,
            "metadata": '{"filename": "sample.pdf", "page_number": 2, "processed_at": "2024-01-01"}'
        }
        
        sample_metadata = {
            "filename": "sample.pdf",
            "page_number": 2,
            "processed_at": "2024-01-01"
        }
        
        # Test source info extraction
        source_info = rag_system._extract_source_info(sample_doc, sample_metadata, 0)
        
        print("✅ Source info extracted:")
        for key, value in source_info.items():
            print(f"   {key}: {value}")
            
        # Verify required fields
        required_fields = [
            'source_id', 'filename', 'title', 'file_path', 'file_url',
            'file_type', 'page_number', 'chunk_index', 'content_preview',
            'relevance_score', 'metadata'
        ]
        
        missing_fields = [field for field in required_fields if field not in source_info]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
        else:
            print("✅ All required fields present in source info")
            
        # Test formatting
        formatted = rag_system.format_sources_for_display([source_info])
        print(f"\n📋 Formatted output:")
        print(formatted)
        
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")

if __name__ == "__main__":
    print("🚀 Source Tracking Verification Script")
    print("This script tests the new source tracking features\n")
    
    test_source_tracking()
    test_source_metadata()
    
    print(f"\n📝 Summary:")
    print("✅ Source tracking adds transparency to RAG responses")
    print("✅ Users can validate AI responses by checking original documents")
    print("✅ Clickable links make it easy to access source documents")
    print("✅ Page numbers help locate exact information in PDFs")
    print("✅ Relevance scores help prioritize which sources to check")
    
    print(f"\n💡 Next steps:")
    print("1. Process your documents using rag_system.initialize_files()")
    print("2. Ask questions and check the sources in the response")
    print("3. Click on file:// links to open documents directly")
    print("4. Use relevance scores to prioritize source verification")
