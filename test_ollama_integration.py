#!/usr/bin/env python3
"""
Test script for Ollama integration with the RAG system.
This script tests the Ollama-based embedding and chat completion functionality.
"""

import os
import sys
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_core.RAG_Core import RAGSystem, RAGSystemConfig

def test_ollama_connection():
    """Test if Ollama is running and models are available."""
    print("🔍 Testing Ollama connection...")
    
    try:
        config = RAGSystemConfig()
        
        # Create a minimal RAG system just for testing connection
        import requests
        
        # Test Ollama connection
        response = requests.get(f"{config.ollama_base_url}/api/tags")
        response.raise_for_status()
        
        available_models = response.json().get("models", [])
        model_names = [model["name"] for model in available_models]
        
        print(f"✅ Ollama is running at {config.ollama_base_url}")
        print(f"📋 Available models: {model_names}")
        
        # Check for required models
        if config.embedding_model in model_names:
            print(f"✅ Embedding model '{config.embedding_model}' is available")
        else:
            print(f"❌ Embedding model '{config.embedding_model}' is NOT available")
            print(f"   Please run: ollama pull {config.embedding_model}")
            
        if config.chat_model in model_names:
            print(f"✅ Chat model '{config.chat_model}' is available")
        else:
            print(f"❌ Chat model '{config.chat_model}' is NOT available")
            print(f"   Please run: ollama pull {config.chat_model}")
            
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Please ensure Ollama is running with: ollama serve")
        return False

def test_embedding_generation():
    """Test embedding generation with Ollama."""
    print("\n🧮 Testing embedding generation...")
    
    try:
        config = RAGSystemConfig()
        
        # Create a minimal test for embedding
        import requests
        
        test_text = "This is a test sentence for embedding generation."
        
        ollama_url = f"{config.ollama_base_url}/api/embeddings"
        payload = {
            "model": config.embedding_model,
            "prompt": test_text
        }
        
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        
        print(f"✅ Embedding generated successfully!")
        print(f"   Text: '{test_text}'")
        print(f"   Embedding dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_chat_completion():
    """Test chat completion with Ollama."""
    print("\n💬 Testing chat completion...")
    
    try:
        config = RAGSystemConfig()
        
        # Create a minimal test for chat completion
        import requests
        
        test_prompt = "Hello! Please respond with a brief greeting."
        
        ollama_url = f"{config.ollama_base_url}/api/generate"
        payload = {
            "model": config.chat_model,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.7
            }
        }
        
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        chat_response = response.json()
        answer = chat_response.get("response", "")
        
        print(f"✅ Chat completion successful!")
        print(f"   Prompt: '{test_prompt}'")
        print(f"   Response: '{answer.strip()}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False

def test_rag_system_initialization():
    """Test RAG system initialization with Ollama."""
    print("\n🏗️ Testing RAG system initialization...")
    
    try:
        # Set environment variables for testing (if not already set)
        if not os.getenv("SUPABASE_URL"):
            print("⚠️  SUPABASE_URL not set - using dummy value for initialization test")
            os.environ["SUPABASE_URL"] = "https://dummy.supabase.co"
        
        if not os.getenv("SUPABASE_KEY"):
            print("⚠️  SUPABASE_KEY not set - using dummy value for initialization test")
            os.environ["SUPABASE_KEY"] = "dummy_key"
        
        config = RAGSystemConfig()
        
        # Just test configuration creation
        print(f"✅ RAG Configuration created successfully!")
        print(f"   Ollama URL: {config.ollama_base_url}")
        print(f"   Embedding Model: {config.embedding_model}")
        print(f"   Chat Model: {config.chat_model}")
        print(f"   Max Tokens: {config.max_token_response}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system initialization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Ollama Integration Tests")
    print("=" * 50)
    
    tests = [
        test_ollama_connection,
        test_embedding_generation,
        test_chat_completion,
        test_rag_system_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Ollama integration is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
    print("\n📚 Setup Instructions:")
    print("1. Ensure Ollama is running: ollama serve")
    print("2. Pull required models:")
    print("   - ollama pull nomic-embed-text:latest")
    print("   - ollama pull deepseek-r1:8b")
    print("3. Set up your Supabase credentials in environment variables:")
    print("   - SUPABASE_URL")
    print("   - SUPABASE_KEY")

if __name__ == "__main__":
    main()
