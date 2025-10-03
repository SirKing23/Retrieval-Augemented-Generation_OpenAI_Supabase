"""
RAG System Terminal Interface
Interactive command-line interface for the RAG system with the new project structure.
"""

import time
import sys
from pathlib import Path
from dotenv import load_dotenv


# Add the rag_core to Python path for importing
current_dir = Path(__file__).parent.absolute()
rag_core_path = current_dir.parent / 'rag_core'
sys.path.insert(0, str(rag_core_path))

from RAG_Core import RAGSystem, RAGSystemConfig



def run_RAG():
   
    print(" RAG System - Terminal Interface ")      
  
    rag_system = RAGSystem()       
    
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
      

def main():   
    print("🚀 Starting Optimized RAG System Examples\n")   

    try:    

        run_RAG()  

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have:")
        print("   - Valid API keys in environment variables")
        print("   - Supabase database properly configured")
        print("   - Documents in the 'uploaded_docs' folder")

if __name__ == "__main__":
    main()
