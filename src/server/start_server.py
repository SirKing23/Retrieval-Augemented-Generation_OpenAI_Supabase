"""
Startup Script for RAG System Web Interface
This script helps you launch the web interface easily
"""

import os
import sys
import subprocess
import time

from pathlib import Path

# Load .env file for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. .env file will not be loaded.")

def check_requirements():
    """Check if required packages are installed"""
    print("Checking system requirements...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'openai',       
        'langchain'
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
    print("\n Checking environment variables...")
    
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")      
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


def start_server():
    """Start the FastAPI server"""
    print("\n Starting server interface...")
    print("=" * 50)    
   
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.server.rag_API:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
        # Wait a moment for server to start
        print("⏳ Starting server...")
        time.sleep(3)
        
        # Open browser
        # import webbrowser
        # url = "http://localhost:8000"
        #  print(f"🌐 Opening web interface at {url}")
        # webbrowser.open(url)
        
        
        print("📱 Web Interface: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/api/docs")
        print("🔧 Alternative Docs: http://localhost:8000/api/redoc")      
        
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
    
    # Make sure we're in the project root directory then Check if RAG_Core.py exists
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)  
    rag_core_path = Path("src/rag_core/RAG_Core.py")

    if not rag_core_path.exists():
        print("❌ Main Module 'RAG_Core' not found!")
        print("Make sure the RAG core module is in the correct location")
        return
    
    # 1.  Check all requirements
    requirements_ok = check_requirements()
    if not requirements_ok:
       print("\n❌ Please install missing requirements first:")       
       return
    
    # 2. Check all environment variables
    env_vars_ok = check_environment_variables() 
    if not env_vars_ok:
        print("\n❌ Please set up environment variables first")
        print("Create a .env file with your API keys")
        return
    
    # 3. Check documents folder

    
    print("\n Initial checks passed!")
    
  
    try:      
        start_server()
      
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled")

if __name__ == "__main__":
    main()
