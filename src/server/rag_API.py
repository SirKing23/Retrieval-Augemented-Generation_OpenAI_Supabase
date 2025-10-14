"""
FastAPI Web Backend for RAG System
Advanced REST API using RAG_Core.py as backend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import time
import uuid
import logging
from pathlib import Path

# Import the RAG system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_core'))
from RAG_Core import RAGSystem

# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask the RAG system")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")

class QuestionResponse(BaseModel):
    response: str
    documents_found: int
    response_time: float
    sources: List[Dict[str, Any]]
    cached: bool
    session_id: str

class UploadRequest(BaseModel):
    directory: str = Field(default="uploaded_docs", description="Directory containing documents to process")

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    processing_time: float

class SystemStats(BaseModel):
    cache_performance: Dict[str, Any]
    api_call_statistics: Dict[str, Any]
    system_health: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="RAG System Web Interface",
    description="Advanced Retrieval-Augmented Generation System with intelligent document search",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
web_interface_path = os.path.join(os.path.dirname(__file__), '..', 'interfaces', 'web_interface')
app.mount("/static", StaticFiles(directory=web_interface_path), name="static")

# Mount documents directory for file access
documents_dir = os.getenv("DOCUMENTS_DIR", "./data/Knowledge_Base_Files")
if os.path.exists(documents_dir):
    app.mount("/documents", StaticFiles(directory=documents_dir), name="documents")

# Global variables
rag_system: Optional[RAGSystem] = None
active_sessions: Dict[str, Dict] = {}
processing_status: Dict[str, Dict] = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rag_system() -> RAGSystem:
    """Get or create RAG system instance"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = RAGSystem()
            logger.info("RAG System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG system: {str(e)}")
    return rag_system

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    try:
        get_rag_system()
        logger.info("FastAPI application started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_system
    if rag_system:
        try:
            rag_system.save_cache()
            logger.info("Cache saved on shutdown")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Serve the main web interface
@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    """Serve the main web application"""
    try:
        html_file_path = os.path.join(os.path.dirname(__file__), '..', 'interfaces', 'web_interface', 'index.html')
        with open(html_file_path, "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: Web interface files not found</h1>", status_code=500)

# API Endpoints
@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Main API endpoint for asking questions"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        # Get RAG system and process question
        rag = get_rag_system()
        
        start_time = time.time()
        response = rag.answer_this(request.question)
        processing_time = time.time() - start_time
        
        # Add session tracking
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "created_at": time.time(),
                "questions_asked": 0,
                "total_response_time": 0
            }
        
        active_sessions[session_id]["questions_asked"] += 1
        active_sessions[session_id]["total_response_time"] += processing_time
        
        # Add session ID to response
        response["session_id"] = session_id
        
        return QuestionResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API call for processing documents in a directory - if the document is already in the processed files,
# it will not be re-processed
@app.post("/api/process-uploaded-files", response_model=UploadResponse)
async def upload_documents(request: UploadRequest, background_tasks: BackgroundTasks):
    """Process documents in specified directory"""
    try:
        directory = request.directory
        
        if not os.path.exists(directory):
            raise HTTPException(status_code=404, detail=f"Directory '{directory}' not found")
        
        rag = get_rag_system()
        
        # Count files before processing
        files_before = len(rag.load_processed_files())
        
        # Process files and measure time
        start_time = time.time()
        rag.initialize_files(directory)
        processing_time = time.time() - start_time
        
        # Count files after processing
        files_after = len(rag.load_processed_files())
        files_processed = files_after - files_before
        
        # Save cache in background
        background_tasks.add_task(rag.save_cache)
        
        return UploadResponse(
            message=f"Successfully processed documents from {directory}",
            files_processed=files_processed,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#API call for uploading a single file to directory without processing for embedding
@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file to the documents directory without processing"""
    try:
        # Get documents directory from environment
        documents_dir = os.getenv("DOCUMENTS_DIR", "./data/Knowledge_Base_Files")
        docs_path = Path(documents_dir)
        
        # Create directory if it doesn't exist
        docs_path.mkdir(parents=True, exist_ok=True)
        
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save file to documents directory
        file_path = docs_path / file.filename
        
        # Check if file already exists
        if file_path.exists():
            # Generate unique filename
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while file_path.exists():
                file_path = docs_path / f"{stem}_{counter}{suffix}"
                counter += 1
        
        # Write file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file_path}")
        
        return {
            "message": f"File '{file.filename}' uploaded successfully",
            "filename": file_path.name,
            "size": len(content),
            "path": str(file_path),
            "processed": False
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def get_files():
    """Get list of processed files"""
    try:
        # Get documents directory from environment
        documents_dir = os.getenv("DOCUMENTS_DIR", "./data/Knowledge_Base_Files")
        docs_path = Path(documents_dir)
        
        files = []
        if docs_path.exists():
            for file_path in docs_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.docx', '.doc']:
                    stat = file_path.stat()
                    files.append({
                        "id": str(file_path.stem),
                        "name": file_path.name,
                        "type": file_path.suffix.lower().replace('.', ''),
                        "size": stat.st_size,
                        "uploaded": time.ctime(stat.st_ctime),
                        "path": str(file_path)
                    })
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x['uploaded'], reverse=True)
        
        return files
    
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=SystemStats)
async def get_statistics():
    """Get comprehensive system performance statistics"""
    try:
        rag = get_rag_system()
        stats = rag.get_performance_report()
        
        # Add session information
        session_stats = {
            "active_sessions": len(active_sessions),
            "total_questions": sum(s["questions_asked"] for s in active_sessions.values()),
            "average_response_time": sum(s["total_response_time"] for s in active_sessions.values()) / max(1, sum(s["questions_asked"] for s in active_sessions.values()))
        }
        
        return SystemStats(
            cache_performance=stats.get("cache_performance", {}),
            api_call_statistics={**stats.get("api_call_statistics", {}), **session_stats},
            system_health={
                "memory_usage": stats.get("memory_usage", {}),
                "cache_storage": stats.get("cache_storage", {}),
                "uptime": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_rag_system()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "rag_system_initialized": rag is not None,
            "active_sessions": len(active_sessions)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "rag_system_initialized": False
        }

@app.delete("/api/cache")
async def clear_cache():
    """Clear all caches"""
    try:
        rag = get_rag_system()
        rag.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_sessions():
    """Get information about active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "questions_asked": info["questions_asked"],
                "created_at": info["created_at"],
                "avg_response_time": info["total_response_time"] / max(1, info["questions_asked"])
            }
            for sid, info in active_sessions.items()
        ]
    }

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    try:
        while True:
            # Receive question from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            question = request_data.get("question", "").strip()
            
            if not question:
                await websocket.send_text(json.dumps({
                    "error": "Question is required"
                }))
                continue
            
            # Process question
            rag = get_rag_system()
            response = rag.answer_this(question)
            response["session_id"] = session_id
            
            # Send response back
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()

#API call for deleting a file in the directory
@app.delete("/api/delete-file/{file_id}")
async def delete_file(file_id: str):
    """Delete a file from the documents directory by file_id"""
    try:
        rag = get_rag_system()
        # Resolve full filename from file_id (stem)
        documents_dir =rag.get_knowledge_base_directory()
        docs_path = Path(documents_dir)
        full_filename = None
        for file_path in docs_path.glob("*"):
            if file_path.is_file() and file_path.stem == file_id:
                full_filename = file_path.name
                break

        if not full_filename:
            raise HTTPException(status_code=404, detail=f"File with id '{file_id}' not found")
        
        isFileDeleted = rag.delete_file_from_documents_dir(full_filename)

        if not isFileDeleted:
            raise HTTPException(status_code=404, detail=f"File '{full_filename}' could not be deleted")
        
        logger.info(f"File deleted successfully: {full_filename}")

        return isFileDeleted
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#API call for deleting embeddings associated with a file
@app.delete("/api/delete-embeddings/{file_id}")
async def delete_embeddings(file_id: str):
    """Delete embeddings associated with a file by file_id (stem)"""
    try:
        rag = get_rag_system()      
        deleted = rag.delete_file_embeddings(file_id)
        if deleted:
            logger.info(f"Embeddings deleted for file_id: {file_id}")
            return {"message": f"Embeddings for file '{file_id}' deleted successfully", "file_id": file_id}
        else:
            raise HTTPException(status_code=404, detail=f"No embeddings found for file id '{file_id}'")
    except Exception as e:
        logger.error(f"Error deleting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
