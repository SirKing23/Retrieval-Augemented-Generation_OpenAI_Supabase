"""
FastAPI Web Backend for RAG System
Advanced REST API with async support, automatic docs, and WebSocket support
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import os
import time
import uuid
from RAG_Core import OptimizedRAGSystem

# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QuestionResponse(BaseModel):
    response: str
    documents_found: int
    response_time: float
    sources: List[dict]
    cached: bool
    session_id: str

class UploadResponse(BaseModel):
    message: str
    files_processed: int

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Advanced Retrieval-Augmented Generation System with caching and performance optimization",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None
active_sessions = {}

def get_rag_system():
    """Get or create RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = OptimizedRAGSystem(use_async=True)
    return rag_system

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    rag_system = OptimizedRAGSystem(use_async=True)
    print("RAG System initialized and ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if rag_system:
        await rag_system.cleanup()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve a simple web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced RAG System</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .chat-section { display: flex; gap: 20px; }
            .chat-container { flex: 2; border: 1px solid #e0e0e0; border-radius: 8px; height: 500px; overflow-y: auto; padding: 15px; background: #fafafa; }
            .stats-container { flex: 1; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background: #f9f9f9; }
            .input-section { margin-top: 20px; display: flex; gap: 10px; }
            .input-section input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
            .input-section button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
            .input-section button:hover { background: #0056b3; }
            .message { margin: 15px 0; padding: 12px; border-radius: 8px; }
            .user { background: #e3f2fd; margin-left: 20%; }
            .assistant { background: #f1f8e9; margin-right: 20%; }
            .sources { font-size: 0.9em; color: #666; margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; }
            .metadata { font-size: 0.8em; color: #888; margin-top: 5px; }
            .stats-item { margin: 10px 0; padding: 8px; background: white; border-radius: 4px; }
            .loading { text-align: center; color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Advanced RAG System</h1>
                <p>Ask questions about your documents with AI-powered search and caching</p>
            </div>
            
            <div class="chat-section">
                <div>
                    <h3>💬 Chat</h3>
                    <div id="chat-container" class="chat-container"></div>
                    <div class="input-section">
                        <input type="text" id="user-input" placeholder="Ask a question about your documents..." onkeypress="handleKeyPress(event)">
                        <button onclick="sendMessage()" id="send-btn">Send</button>
                    </div>
                </div>
                
                <div>
                    <h3>📊 Performance Stats</h3>
                    <div id="stats-container" class="stats-container">
                        <div class="loading">Loading stats...</div>
                    </div>
                    <button onclick="refreshStats()" style="width: 100%; margin-top: 10px; padding: 8px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Refresh Stats</button>
                </div>
            </div>
        </div>

        <script>
            let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            
            async function sendMessage() {
                const input = document.getElementById('user-input');
                const sendBtn = document.getElementById('send-btn');
                const question = input.value.trim();
                if (!question) return;

                // Update UI
                addMessage(question, 'user');
                input.value = '';
                sendBtn.disabled = true;
                sendBtn.textContent = 'Processing...';

                try {
                    const response = await fetch('/api/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            question: question,
                            session_id: sessionId 
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        addMessage(`❌ Error: ${data.error}`, 'assistant');
                    } else {
                        addMessage(data.response, 'assistant', data.sources, data);
                        refreshStats(); // Update stats after each query
                    }
                } catch (error) {
                    addMessage(`❌ Network Error: ${error.message}`, 'assistant');
                } finally {
                    sendBtn.disabled = false;
                    sendBtn.textContent = 'Send';
                }
            }

            function addMessage(text, sender, sources = null, metadata = null) {
                const chatContainer = document.getElementById('chat-container');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                let content = `<div><strong>${sender === 'user' ? '👤 You' : '🤖 Assistant'}:</strong></div><div>${text}</div>`;
                
                if (sources && sources.length > 0) {
                    content += '<div class="sources"><strong>📚 Sources:</strong><ul style="margin: 5px 0; padding-left: 20px;">';
                    sources.slice(0, 3).forEach(source => {
                        content += `<li><strong>${source.title}</strong> (${(source.relevance_score * 100).toFixed(1)}% relevant)`;
                        if (source.page_number) content += ` - Page ${source.page_number}`;
                        content += `</li>`;
                    });
                    if (sources.length > 3) content += `<li><em>... and ${sources.length - 3} more sources</em></li>`;
                    content += '</ul></div>';
                }
                
                if (metadata && sender === 'assistant') {
                    content += `<div class="metadata">
                        ⏱️ ${metadata.response_time?.toFixed(2)}s | 
                        📄 ${metadata.documents_found} docs | 
                        ${metadata.cached ? '💾 Cached' : '🔄 Fresh'} |
                        Session: ${metadata.session_id?.substring(0, 8)}...
                    </div>`;
                }
                
                messageDiv.innerHTML = content;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            async function refreshStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    const container = document.getElementById('stats-container');
                    container.innerHTML = `
                        <div class="stats-item">
                            <strong>💾 Cache Performance</strong><br>
                            Embedding: ${stats.cache_performance.embedding_cache.hit_rate_percentage}% hit rate<br>
                            Response: ${stats.cache_performance.response_cache.hit_rate_percentage}% hit rate
                        </div>
                        <div class="stats-item">
                            <strong>🔌 API Calls</strong><br>
                            Total: ${stats.api_call_statistics.total_openai_calls}<br>
                            Saved: ${stats.cache_performance.overall_cache_efficiency.total_api_calls_saved}
                        </div>
                        <div class="stats-item">
                            <strong>💬 Session</strong><br>
                            History: ${stats.chat_history_length} messages<br>
                            Cache Size: ${stats.embedding_cache_size} items
                        </div>
                        <div class="stats-item">
                            <strong>💾 Storage</strong><br>
                            Cache: ${(stats.cache_storage.total_cache_size_mb || 0).toFixed(1)} MB<br>
                            Items: ${(stats.cache_storage.embedding_cache_items || 0) + (stats.cache_storage.response_cache_items || 0)}
                        </div>
                    `;
                } catch (error) {
                    console.error('Error refreshing stats:', error);
                }
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Initialize
            refreshStats();
            
            // Auto-refresh stats every 30 seconds
            setInterval(refreshStats, 30000);
        </script>
    </body>
    </html>
    """

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Main API endpoint for asking questions"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        # Get RAG system and process question
        rag = get_rag_system()
        response = rag.answer_this(request.question)
        
        # Add session ID to response
        response["session_id"] = session_id
        
        return QuestionResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_documents(directory: str = "uploaded_docs"):
    """Process documents in specified directory"""
    try:
        if not os.path.exists(directory):
            raise HTTPException(status_code=404, detail="Upload directory not found")
        
        rag = get_rag_system()
        
        # Count files before processing
        files_before = len(rag.load_processed_files())
        
        # Process files
        rag.initialize_files(directory)
        
        # Count files after processing
        files_after = len(rag.load_processed_files())
        files_processed = files_after - files_before
        
        return UploadResponse(
            message=f"Documents processed successfully",
            files_processed=files_processed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """Get comprehensive system performance statistics"""
    try:
        rag = get_rag_system()
        stats = rag.get_performance_report()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "rag_system_initialized": rag_system is not None
    }

@app.delete("/api/cache")
async def clear_cache():
    """Clear all caches"""
    try:
        rag = get_rag_system()
        rag.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_sessions[session_id] = websocket
    
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
        if session_id in active_sessions:
            del active_sessions[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
