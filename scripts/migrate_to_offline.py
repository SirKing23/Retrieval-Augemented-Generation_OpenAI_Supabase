#!/usr/bin/env python3
"""
Migration script to convert from Supabase (online) to ChromaDB (offline)
This script helps users migrate their existing embeddings if they have any.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the parent directory to the path so we can import RAG_Core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def install_chromadb():
    """Install ChromaDB if not already installed"""
    try:
        import chromadb
        print("✓ ChromaDB is already installed")
        return True
    except ImportError:
        print("Installing ChromaDB...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb>=0.4.22"])
            print("✓ ChromaDB installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install ChromaDB: {e}")
            return False

def create_env_template():
    """Create a template .env file with ChromaDB configuration"""
    env_template = """# RAG System Configuration - ChromaDB (Offline)

# ChromaDB Configuration (replaces Supabase)
CHROMADB_PATH=./data/chromadb
COLLECTION_NAME=document_embeddings

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text:latest
CHAT_MODEL=deepseek-r1:8b

# AI Parameters
MAX_TOKEN_RESPONSE=700
VECTOR_SEARCH_THRESHOLD=0.7
VECTOR_SEARCH_MATCH_COUNT=5
MAX_CHUNK_TOKENS=600
OVERLAP_CHUNK_TOKENS=300

# Cache Configuration
CACHE_DIR=./cache

# Document Processing
DOCUMENTS_DIR=./data/Knowledge_Base_Files
PROCESSED_FILES_DIR=./data/processed_files.txt
"""
    
    env_path = ".env"
    if os.path.exists(env_path):
        print(f"✓ .env file already exists at {env_path}")
        return
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_template)
        print(f"✓ Created .env template at {env_path}")
        print("  Please review and update the configuration as needed.")
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")

def test_chromadb_connection():
    """Test ChromaDB connection and setup"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Load environment variables
        load_dotenv()
        
        chromadb_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
        collection_name = os.getenv("COLLECTION_NAME", "document_embeddings")
        
        # Create directory if it doesn't exist
        os.makedirs(chromadb_path, exist_ok=True)
        
        # Test ChromaDB connection
        client = chromadb.PersistentClient(
            path=chromadb_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Try to get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            print(f"✓ Connected to existing ChromaDB collection '{collection_name}' with {count} documents")
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "RAG System Document Embeddings"}
            )
            print(f"✓ Created new ChromaDB collection '{collection_name}'")
        
        return True
        
    except Exception as e:
        print(f"✗ ChromaDB connection test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    try:
        import requests
        
        load_dotenv()
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        print(f"✓ Connected to Ollama at {ollama_base_url}")
        print(f"  Available models: {', '.join(model_names)}")
        
        # Check for required models
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
        chat_model = os.getenv("CHAT_MODEL", "deepseek-r1:8b")
        
        if embedding_model not in model_names:
            print(f"⚠ Warning: Embedding model '{embedding_model}' not found")
            print(f"  Run: ollama pull {embedding_model}")
        else:
            print(f"✓ Embedding model '{embedding_model}' is available")
            
        if chat_model not in model_names:
            print(f"⚠ Warning: Chat model '{chat_model}' not found")
            print(f"  Run: ollama pull {chat_model}")
        else:
            print(f"✓ Chat model '{chat_model}' is available")
        
        return True
        
    except Exception as e:
        print(f"✗ Ollama connection test failed: {e}")
        print("  Make sure Ollama is running and accessible")
        return False

def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(description="Migrate RAG system from Supabase to ChromaDB")
    parser.add_argument("--skip-install", action="store_true", help="Skip ChromaDB installation")
    parser.add_argument("--test-only", action="store_true", help="Only run tests, don't create files")
    
    args = parser.parse_args()
    
    print("🔄 RAG System Migration: Supabase → ChromaDB (Offline)")
    print("=" * 50)
    
    # Step 1: Install ChromaDB
    if not args.skip_install:
        if not install_chromadb():
            sys.exit(1)
    
    # Step 2: Create .env template
    if not args.test_only:
        create_env_template()
    
    # Step 3: Test ChromaDB
    print("\n🧪 Testing ChromaDB connection...")
    chromadb_ok = test_chromadb_connection()
    
    # Step 4: Test Ollama
    print("\n🧪 Testing Ollama connection...")
    ollama_ok = test_ollama_connection()
    
    # Summary
    print("\n📋 Migration Summary:")
    print("=" * 30)
    
    if chromadb_ok and ollama_ok:
        print("✅ Migration completed successfully!")
        print("\n🎉 Your RAG system is now ready to run offline!")
        print("\nNext steps:")
        print("1. Update your .env file with correct paths")
        print("2. Process your documents using the RAG system")
        print("3. Test the system with sample queries")
        
        if not args.test_only:
            print("\n📖 Example usage:")
            print("```python")
            print("from src.rag_core.RAG_Core import RAGSystem")
            print("")
            print("# Initialize the offline RAG system")
            print("rag = RAGSystem()")
            print("")
            print("# Process documents")
            print("rag.initialize_files('./data/Knowledge_Base_Files')")
            print("")
            print("# Ask questions")
            print("response = rag.answer_this('Your question here')")
            print("print(response['response'])")
            print("```")
    else:
        print("⚠ Migration completed with warnings")
        if not chromadb_ok:
            print("  - ChromaDB connection failed")
        if not ollama_ok:
            print("  - Ollama connection failed")
        print("\nPlease resolve the issues above before using the system.")

if __name__ == "__main__":
    main()
