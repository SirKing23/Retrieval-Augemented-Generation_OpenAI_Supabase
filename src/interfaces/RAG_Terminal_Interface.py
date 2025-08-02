"""
RAG System Terminal Interface
Interactive command-line interface for the RAG system with the new project structure.
"""

import asyncio
import os
import time
import sys
from pathlib import Path

# Add the rag_core to Python path for importing
current_dir = Path(__file__).parent.absolute()
rag_core_path = current_dir.parent / 'rag_core'
sys.path.insert(0, str(rag_core_path))

from RAG_Core import RAGSystem, RAGSystemConfig

def find_documents_folder():
    """Find a suitable documents folder to process"""
    possible_paths = [
        "C:\\Users\\YourKing\\Desktop\\RAG_File_Upload",  # Default folder name
        "documents",      # Alternative name
        "docs",          # Another alternative
        "data",          # If using data folder
        "../../data",    # Relative to interface location
    ]
    
    # Check each possible path
    for path in possible_paths:
        abs_path = Path(path).resolve()
        if abs_path.exists() and abs_path.is_dir():
            # Check if it contains any files
            files = list(abs_path.glob('*'))
            if files:
                return str(abs_path)
    
    return None

def basic_usage_example():
    """Basic usage example with dynamic document discovery"""
    print("🚀 RAG System Basic Usage Demo")
    
    # Find documents folder automatically
    documents_path = find_documents_folder()
    if not documents_path:
        print("⚠️ No documents folder found. Creating 'uploaded_docs' folder...")
        documents_path = "uploaded_docs"
        os.makedirs(documents_path, exist_ok=True)
        print(f"📁 Created folder: {documents_path}")
        print("📝 Please add some .txt, .pdf, or .docx files to this folder and run again.")
        return
    
    print(f"📁 Using documents from: {documents_path}")
    
    # Check if folder has any documents
    supported_extensions = ['.txt', '.pdf', '.docx']
    docs_found = []
    for ext in supported_extensions:
        docs_found.extend(Path(documents_path).glob(f'*{ext}'))
    
    if not docs_found:
        print(f"⚠️ No supported documents found in {documents_path}")
        print(f"📝 Please add files with extensions: {', '.join(supported_extensions)}")
        return
    
    print(f"📚 Found {len(docs_found)} documents to process")
    
    # Initialize the optimized RAG system
    # It will automatically load configuration from environment variables
    rag_system = RAGSystem()
    
    # Process some documents
    try:
        rag_system.initialize_files(documents_path)
        print("✅ Documents processed successfully")
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
    
    # Interactive conversation loop
    print("\n💬 Starting interactive conversation mode...")
    print("Type 'exit' to quit the program")
    print("=" * 50)
    
    while True:
        # Get user input
        question = input("\n🤔 Your question: ").strip()
        
        # Check if user wants to exit
        if question.lower() == 'exit':
            print("\n👋 Goodbye! Thanks for using the RAG system!")
            break
        
        # Skip empty questions
        if not question:
            print("❓ Please enter a question or type 'exit' to quit.")
            continue
        
        # Process the question
        print(f"\n🔍 Searching for information about: {question}")
        start_time = time.time()
        response = rag_system.answer_this(question)
        end_time = time.time()
        
        print(f"\n🤖 Answer: {response['response']}")
        print(f"📊 Documents found: {response.get('documents_found', 0)}")
        print(f"⏱️ Response time: {end_time - start_time:.2f} seconds")
        
        # Display source documents
        sources = response.get('sources', [])
        if sources:
            print(f"\n📚 Source Documents ({len(sources)} found):")
            for i, source in enumerate(sources, 1):
                print(f"\n{i}. **{source['title']}** ({source['file_type']})")
                if source['page_number']:
                    print(f"   📄 Page: {source['page_number']}")
                if source['relevance_score'] > 0:
                    print(f"   🎯 Relevance: {source['relevance_score']:.1%}")
                if source['file_url']:
                    print(f"   🔗 Link: {source['file_url']}")
                if source['content_preview']:
                    print(f"   📝 Preview: {source['content_preview']}")
        else:
            print("\n📚 No source documents found")
        
        # Show performance stats periodically
        performance_report = rag_system.get_performance_report()
        cache_perf = performance_report['cache_performance']
        
        print(f"\n📈 Session Stats:")
        print(f"   💾 Cache hits: {cache_perf['embedding_cache']['hits']} | Response cache: {cache_perf['response_cache']['hits']}")
        print(f"   🔌 API calls this session: {performance_report['api_call_statistics']['total_openai_calls']}")
        print(f"   💬 Questions asked: {performance_report['chat_history_length']}")
        
        print("\n" + "=" * 50)

def advanced_configuration_example():
    """Example with custom configuration"""
    print("\n\n=== Advanced Configuration Example ===")
    
    # Create custom configuration
    config = RAGSystemConfig()
    
    # Override some settings
    config.vector_search_match_count = 5  # Get more documents
    config.max_token_response = 1000      # Longer responses
    config.vector_search_threshold = 0.7  # Higher similarity threshold
    
    # Initialize with custom config
    rag_system = RAGSystem(config=config)
    
    # Test with a different question
    question = "How to use GroupLoadByName? also give example of how to use it."
    print(f"\n🤔 Question: {question}")
    
    response = rag_system.answer_this(question)
    print(f"\n🤖 Answer: {response['response']}")
    print(f"📊 Documents found: {response.get('documents_found', 0)}")
    
    # Show sources with higher match count
    sources = response.get('sources', [])
    if sources:
        print(f"\n📚 Sources (with higher match count):")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source['title']} - Relevance: {source['relevance_score']:.1%}")
            if source['file_url']:
                print(f"   🔗 {source['file_url']}")

def source_validation_example():
    """Example showing how users can validate responses using sources"""
    print("\n\n=== Source Validation Example ===")
    
    rag_system = RAGSystem()
    
    question = "How to use GroupLoadByName? also give example of how to use it."
    print(f"\n🤔 Question: {question}")
    
    response = rag_system.answer_this(question)
    print(f"\n🤖 Answer: {response['response']}")
    
    sources = response.get('sources', [])
    if sources:
        print(f"\n🔍 Source Validation Information:")
        print(f"Found {len(sources)} relevant documents to validate this response:")
        
        for i, source in enumerate(sources, 1):
            print(f"\n📄 Source {i}:")
            print(f"   📁 Document: {source['filename']}")
            if source['page_number']:
                print(f"   📄 Page: {source['page_number']}")
            print(f"   🎯 Relevance Score: {source['relevance_score']:.1%}")
            print(f"   📝 Content Preview: {source['content_preview']}")
            
            if source['file_url']:
                print(f"   🔗 Click to open: {source['file_url']}")
                print(f"   💡 You can verify the AI's response by checking this document")
            elif source['file_path']:
                print(f"   📁 File location: {source['file_path']}")
        
        print(f"\n✅ Recommendation: Check the highest relevance sources first")
        print(f"📈 Sources are ranked by relevance to your question")
    else:
        print("\n⚠️  No sources found - response may be from general knowledge")

def batch_processing_example():
    """Example of batch processing multiple questions"""
    print("\n\n=== Batch Processing Example ===")
    
    rag_system = RAGSystem()
    
    questions = [
        "What is ProductCenter?",
        "How do I install it?",
        "What are the system requirements?",
        "How do I configure the database?"
    ]
    
    print("🔄 Processing multiple questions...")
    
    total_start = time.time()
    responses = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        response = rag_system.answer_this(question)
        responses.append(response)
        print(f"✅ Answered in {response.get('response_time', 0):.2f}s")
    
    total_time = time.time() - total_start
    print(f"\n⏱️ Total batch processing time: {total_time:.2f} seconds")
    print(f"📊 Average time per question: {total_time/len(questions):.2f} seconds")

def error_handling_example():
    """Example demonstrating error handling and recovery"""
    print("\n\n=== Error Handling Example ===")
    
    rag_system = RAGSystem()
    
    # Test with invalid input
    invalid_inputs = [
        "",  # Empty string
        None,  # None value
        "A" * 15000,  # Very long string
        "<script>alert('test')</script>",  # Potential injection
    ]
    
    for i, invalid_input in enumerate(invalid_inputs, 1):
        try:
            print(f"\n🧪 Test {i}: Testing invalid input...")
            response = rag_system.answer_this(str(invalid_input) if invalid_input else "")
            if 'error' in response:
                print(f"✅ Error handled gracefully: {response.get('error', 'Unknown error')}")
            else:
                print(f"✅ Input processed successfully")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def caching_demonstration():
    """Demonstrate the caching functionality"""
    print("\n\n=== Caching Demonstration ===")
    
    rag_system = RAGSystem()
    
    question = "What is ProductCenter?"
    
    # First call (no cache)
    print("🔄 First call (no cache)...")
    start_time = time.time()
    response1 = rag_system.answer_this(question)
    time1 = time.time() - start_time
    print(f"⏱️ Time: {time1:.2f} seconds")
    
    # Second call (with cache)
    print("\n🔄 Second call (with cache)...")
    start_time = time.time()
    response2 = rag_system.answer_this(question)
    time2 = time.time() - start_time
    print(f"⏱️ Time: {time2:.2f} seconds")
    
    print(f"\n📈 Performance improvement: {((time1 - time2) / time1 * 100):.1f}% faster")
    
    # Show cache statistics
    performance_report = rag_system.get_performance_report()
    print(f"💾 Cache size: {performance_report['embedding_cache_size']} embeddings")

def memory_management_example():
    """Demonstrate memory management features"""
    print("\n\n=== Memory Management Example ===")
    
    rag_system = RAGSystem()
    
    # Simulate long conversation to test chat history management
    print("🗣️ Simulating long conversation...")
    
    for i in range(25):  # More than max_history_length (20)
        question = f"This is question number {i+1} about ProductCenter."
        response = rag_system.answer_this(question)
        
        if i % 5 == 0:
            print(f"   Question {i+1}: Chat history length = {len(rag_system.chat_history)}")
    
    print(f"\n📊 Final chat history length: {len(rag_system.chat_history)}")
    print("✅ Chat history was automatically managed to prevent memory overflow")
    
    # Show memory usage
    performance_report = rag_system.get_performance_report()
    print(f"\n💾 Memory usage:")
    for key, value in performance_report['memory_usage'].items():
        print(f"   {key}: {value} bytes")

async def async_example():
    """Example of async operations (if enabled)"""
    print("\n\n=== Async Operations Example ===")
    
    # Create async-enabled system
    rag_system = RAGSystem(use_async=True)
    
    questions = [
        "What is ProductCenter?",
        "How do I install it?",
        "What are the requirements?"
    ]
    
    print("🚀 Processing questions asynchronously...")
    
    # Process questions concurrently
    start_time = time.time()
    tasks = []
    
    for question in questions:
        # Note: In this simplified example, we're not fully async
        # In a real implementation, you'd have true async HTTP calls
        task = rag_system.generate_embedding_async(question)
        tasks.append(task)
    
    # Wait for all embeddings to complete
    embeddings = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"✅ Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
    
    # Cleanup
    await rag_system.cleanup()

def main():
    """Run all examples"""
    print("🚀 Starting Optimized RAG System Examples\n")
    
    try:
        # Basic examples
        basic_usage_example()
       # advanced_configuration_example()
       # source_validation_example()
        # batch_processing_example()
        
        # # Advanced features
        # error_handling_example()
        # caching_demonstration()
        # memory_management_example()
        
        # # Async example (commented out by default)
        # # asyncio.run(async_example())
                
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have:")
        print("   - Valid API keys in environment variables")
        print("   - Supabase database properly configured")
        print("   - Documents in the 'uploaded_docs' folder")

if __name__ == "__main__":
    main()
