"""
Startup Script for RAG System Web Interface
This script helps you launch the web interface easily
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# Load .env file for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. .env file will not be loaded.")

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'supabase',
        'langchain',
        'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r ../../requirements/web_requirements.txt")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n🔍 Checking environment variables...")
    
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"), 
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        "CHAT_MODEL": os.getenv("CHAT_MODEL", "deepseek-r1:8b")
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if var_value:
            print(f"✅ {var_name} - set")
        else:
            missing_vars.append(var_name)
            print(f"❌ {var_name} - missing")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Create a .env file with these variables or set them in your system")
        return False
    print("✅ All environment variables set!")
    return True

def check_ollama_server_and_models():
    """Check if Ollama server is running and required models are available"""
    print("\n🔍 Checking Ollama server and models...")
    import requests
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    chat_model = os.getenv("CHAT_MODEL", "deepseek-r1:8b")
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        if embedding_model not in model_names:
            print(f"❌ Embedding model '{embedding_model}' not found in Ollama. Run: ollama pull {embedding_model}")
            return False
        if chat_model not in model_names:
            print(f"❌ Chat model '{chat_model}' not found in Ollama. Run: ollama pull {chat_model}")
            return False
        print(f"✅ Ollama server running. Models available: {embedding_model}, {chat_model}")
        return True
    except Exception as e:
        print(f"❌ Ollama server not running or unreachable at {ollama_base_url}. Error: {e}")
        print("Start Ollama with: ollama serve")
        return False

def check_documents_folder():
    """Check if documents folder exists"""
    print("\n🔍 Checking documents folder...")
    
    # Get documents directory from environment variable
    documents_dir = os.getenv("DOCUMENTS_DIR", "./data/Knowledge_Base_Files")
    docs_folder = Path(documents_dir)
    
    print(f"📁 Looking for documents in: {docs_folder}")
    
    if docs_folder.exists():
        doc_files = list(docs_folder.glob("*.pdf")) + list(docs_folder.glob("*.txt")) + list(docs_folder.glob("*.docx"))
        if doc_files:
            print(f"✅ {docs_folder} folder exists with {len(doc_files)} documents")
            print("📄 Documents found:")
            for doc in doc_files[:5]:  # Show first 5
                print(f"   - {doc.name}")
            if len(doc_files) > 5:
                print(f"   ... and {len(doc_files) - 5} more")
        else:
            print(f"⚠️ {docs_folder} folder exists but no documents found")
            print("📄 Add your documents (PDF, TXT, DOCX) to this folder")
    else:
        print(f"❌ Documents folder not found: {docs_folder}")
        print("Creating documents folder...")
        docs_folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created documents folder: {docs_folder}")
        print("📄 Add your documents (PDF, TXT, DOCX) to this folder")

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting RAG System Web Interface...")
    print("=" * 50)
    
    # Make sure we're in the project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    print(f"📁 Working directory: {project_root}")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.server.rag_webapp:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
        # Wait a moment for server to start
        print("⏳ Starting server...")
        time.sleep(3)
        
        # Open browser
        url = "http://localhost:8000"
        print(f"🌐 Opening web interface at {url}")
        webbrowser.open(url)
        
        print("\n✅ RAG System Web Interface is running!")
        print("=" * 50)
        print("📱 Web Interface: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/api/docs")
        print("🔧 Alternative Docs: http://localhost:8000/api/redoc")
        print("\n💡 Tips:")
        documents_dir = os.getenv("DOCUMENTS_DIR", "./data/Knowledge_Base_Files")
        print(f"   - Upload documents to the '{documents_dir}' folder")
        print("   - Use the 'Process Documents' button to index new files")
        print("   - Ask questions about your documents in the chat")
        print("   - Monitor performance in the sidebar")
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        process.terminate()
        print("✅ Server stopped successfully")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Make sure all requirements are installed and try again")

def main():
    """Main startup function"""
    print("🤖 RAG System Web Interface Startup")
    print("=" * 50)
    
    # Make sure we're in the project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Check if RAG_Core.py exists
    rag_core_path = Path("src/rag_core/RAG_Core.py")
    if not rag_core_path.exists():
        print("❌ RAG_Core.py not found!")
        print("Make sure the RAG core module is in the correct location")
        return
    
    # Check all requirements
    requirements_ok = check_requirements()
    env_vars_ok = check_environment_variables()
    ollama_ok = check_ollama_server_and_models()
    check_documents_folder()

    if not requirements_ok:
        print("\n❌ Please install missing requirements first:")
        print("pip install -r requirements/web_requirements.txt")
        return

    if not env_vars_ok:
        print("\n❌ Please set up environment variables first")
        print("Create a .env file with your Supabase and Ollama settings")
        return

    if not ollama_ok:
        print("\n❌ Please start Ollama and pull the required models before continuing.")
        print("See OLLAMA_SETUP_GUIDE.md for instructions.")
        return

    print("\n🎉 All checks passed! Ready to start the web interface.")

    try:
        start_server()
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled")

if __name__ == "__main__":
    main()
