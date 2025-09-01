#!/usr/bin/env python3
"""
Test script to verify the document relevance assessment functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'rag_core'))

from RAG_Core import RAGSystem

def test_relevance_assessment():
    """Test the new relevance assessment functionality"""
    
    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag = RAGSystem()
        
        # Test question that should not use documents from knowledge base
        test_question = "which postgres database is better? weaviate or supabase?"
        
        print(f"\nTesting query: '{test_question}'")
        print("=" * 50)
        
        # Get response
        response = rag.answer_this(test_question)
        
        # Print results
        print(f"Route used: {response.get('route_used', 'unknown')}")
        print(f"Documents found: {response.get('documents_found', 0)}")
        print(f"Sources included: {response.get('sources_included', 'unknown')}")
        print(f"Number of sources: {len(response.get('sources', []))}")
        
        if 'classification' in response:
            classification = response['classification']
            print(f"Classification: {classification.get('category', 'unknown')} (confidence: {classification.get('confidence', 0):.2f})")
        
        if 'relevance_analysis' in response:
            relevance = response['relevance_analysis']
            print(f"Relevance score: {relevance.get('relevance_score', 0):.2f}")
            print(f"Assessment reason: {relevance.get('assessment_reason', 'unknown')}")
            print(f"Should include sources: {relevance.get('should_include_sources', 'unknown')}")
        
        print(f"\nResponse: {response.get('response', 'No response')[:200]}...")
        
        # Test routing statistics
        print("\nRouting Statistics:")
        stats = rag.routing_stats
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing RAG System Relevance Assessment")
    print("=" * 50)
    
    success = test_relevance_assessment()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
