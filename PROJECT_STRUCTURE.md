# RAG System - Project Structure

## 📁 Project Organization

```
📦 Retrieval-Augmented-Generation/
├── 📁 src/                          # Source code
│   ├── 📁 rag_core/                 # RAG System core logic
│   │   └── 📄 RAG_Core.py           # Main RAG implementation
│   ├── 📁 server/                   # Web server components
│   │   ├── 📄 rag_webapp.py         # FastAPI web application
│   │   └── 📄 start_website.py      # Server startup script
│   └── 📁 interfaces/               # User interfaces
│       ├── 📄 RAG_Terminal_Interface.py  # CLI interface
│       └── 📁 web_interface/        # Web UI files
│           ├── 📄 index.html        # Main HTML page
│           ├── 📄 styles.css        # CSS styling
│           └── 📄 app.js            # JavaScript functionality
├── 📁 cache/                        # Caching system
│   └── 📁 rag_cache/               # RAG system cache files
│       ├── 📄 cache_metadata.json  # Cache metadata
│       ├── 📄 embedding_cache.pkl  # Cached embeddings
│       └── 📄 response_cache.pkl   # Cached responses
├── 📁 data/                         # Data and logs
│   ├── 📄 processed_files.txt      # List of processed files
│   └── 📄 rag_system.log          # System logs
├── 📁 requirements/                 # Dependencies
│   ├── 📄 requirements_optimized.txt # Core dependencies
│   └── 📄 web_requirements.txt     # Web interface dependencies
├── 📁 docs/                         # Documentation
│   └── 📁 guidelines/              # Development guidelines
│       ├── 📄 IMPLEMENTATION_GUIDE.md
│       ├── 📄 MIGRATION_GUIDE.md
│       ├── 📄 RAG_Optimization_Report.md
│       ├── 📄 SOURCE_TRACKING_GUIDE.md
│       ├── 📄 Steps for Improving the RAG.txt
│       ├── 📄 WEB_INTERFACE_README.md
│       └── 📄 note.txt
├── 📁 scripts/                      # Utility scripts and demos
│   ├── 📁 demo_files/              # Demo and test scripts
│   │   ├── 📄 cache_performance_demo.py
│   │   ├── 📄 cache_persistence_demo.py
│   │   ├── 📄 persistent_cache_demo.py
│   │   ├── 📄 test_cache_fix.py
│   │   ├── 📄 test_optimized_rag.py
│   │   ├── 📄 test_persistent_cache.py
│   │   └── 📄 test_source_tracking.py
│   └── 📁 deprecated/              # Legacy code
│       └── 📄 main_rag.py
├── 📄 .env                         # Environment variables
├── 📄 .gitignore                   # Git ignore rules
├── 📄 README.md                    # Main project README
└── 📄 PROJECT_STRUCTURE.md         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements/requirements_optimized.txt

# Install web interface dependencies  
pip install -r requirements/web_requirements.txt
```

### 2. Set up Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
OPENAI_API_KEY=your_api_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 3. Run the Application

#### Option A: Web Interface (Recommended)
```bash
cd src/server
python start_website.py
```

#### Option B: Terminal Interface
```bash
cd src/interfaces
python RAG_Terminal_Interface.py
```

#### Option C: Direct FastAPI Server
```bash
cd src/server
uvicorn rag_webapp:app --host 0.0.0.0 --port 8000 --reload
```

## 📂 Folder Descriptions

### `/src/` - Source Code
Contains all the main application source code organized by functionality.

- **`rag_core/`** - Core RAG system implementation
- **`server/`** - Web server and API components  
- **`interfaces/`** - User interface implementations

### `/cache/` - Caching System
Stores all cache-related files for performance optimization.

### `/data/` - Data and Logs
Contains processed file lists, system logs, and other data files.

### `/requirements/` - Dependencies
All dependency files organized by use case.

### `/docs/` - Documentation
Project documentation, guidelines, and development notes.

### `/scripts/` - Utilities and Demos
Demo scripts, testing utilities, and deprecated code.

## 🔄 Migration Notes

If you're updating from the old structure:

1. **Import Paths**: Updated to use relative imports
2. **File Paths**: Static files and data references updated
3. **Scripts**: Demo and utility scripts moved to `/scripts/`
4. **Dependencies**: Separated into focused requirement files

## 🛠️ Development

### Adding New Features
- Core logic: Add to `/src/rag_core/`
- API endpoints: Extend `/src/server/rag_webapp.py`
- UI components: Modify files in `/src/interfaces/web_interface/`

### Testing
- Run demo scripts from `/scripts/demo_files/`
- Check logs in `/data/rag_system.log`
- Monitor cache performance in `/cache/rag_cache/`

This structure provides better organization, clearer separation of concerns, and easier maintenance of the RAG system.
