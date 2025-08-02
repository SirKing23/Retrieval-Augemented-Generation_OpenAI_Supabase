let sessionId = 'session_' + Math.random().toString(36).substr(2, 12);
let messageCounter = 0;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    refreshStats();
    checkSystemHealth();
    setInterval(refreshStats, 30000); // Auto-refresh every 30 seconds
    setInterval(checkSystemHealth, 60000); // Check health every minute
});

async function sendMessage() {
    const input = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const sendIcon = document.getElementById('send-icon');
    const sendText = document.getElementById('send-text');
    
    const question = input.value.trim();
    if (!question) return;

    // Update UI
    addMessage(question, 'user');
    input.value = '';
    
    // Disable send button and show loading
    sendBtn.disabled = true;
    sendIcon.textContent = '⏳';
    sendText.textContent = 'Processing...';

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
        
        if (response.ok) {
            addMessage(data.response, 'assistant', data.sources, data);
            refreshStats(); // Update stats after each query
        } else {
            addMessage(`❌ Error: ${data.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        addMessage(`❌ Network Error: ${error.message}`, 'error');
    } finally {
        // Reset send button
        sendBtn.disabled = false;
        sendIcon.textContent = '📤';
        sendText.textContent = 'Send';
    }
}

function addMessage(text, type, sources = null, metadata = null) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.id = `message-${++messageCounter}`;
    
    let headerIcon = '';
    let headerText = '';
    
    switch(type) {
        case 'user':
            headerIcon = '👤';
            headerText = 'You';
            break;
        case 'assistant':
            headerIcon = '🤖';
            headerText = 'Assistant';
            break;
        case 'error':
            headerIcon = '⚠️';
            headerText = 'Error';
            break;
    }
    
    let content = `
        <div class="message-header">
            ${headerIcon} ${headerText}
        </div>
        <div class="message-content">${text}</div>
    `;
    
    if (sources && sources.length > 0) {
        content += '<div class="sources"><h4>📚 Sources Found:</h4>';
        sources.slice(0, 5).forEach((source, index) => {
            content += `
                <div class="source-item">
                    <strong>${source.title || source.filename || 'Unknown Document'}</strong>
                    ${source.relevance_score ? `<span style="color: #10b981; font-weight: 600;"> (${(source.relevance_score * 100).toFixed(1)}% relevant)</span>` : ''}
                    ${source.page_number ? `<br><small>📄 Page ${source.page_number}</small>` : ''}
                    ${source.content_preview ? `<br><small style="color: #6b7280;">${source.content_preview}</small>` : ''}
                </div>
            `;
        });
        if (sources.length > 5) {
            content += `<div class="source-item"><em>... and ${sources.length - 5} more sources</em></div>`;
        }
        content += '</div>';
    }
    
    if (metadata && type === 'assistant') {
        content += `
            <div class="metadata">
                <span>⏱️ ${metadata.response_time?.toFixed(2)}s</span>
                <span>📄 ${metadata.documents_found} docs</span>
                <span>${metadata.cached ? '💾 Cached' : '🔄 Fresh'}</span>
                <span>🎯 Session: ${metadata.session_id?.substring(0, 8)}...</span>
            </div>
        `;
    }
    
    messageDiv.innerHTML = content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function refreshStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        const container = document.getElementById('stats-container');
        const cachePerf = stats.cache_performance || {};
        const apiStats = stats.api_call_statistics || {};
        
        container.innerHTML = `
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">${cachePerf.embedding_cache?.hit_rate_percentage || 0}%</div>
                    <div class="stat-label">Embedding Cache</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${cachePerf.response_cache?.hit_rate_percentage || 0}%</div>
                    <div class="stat-label">Response Cache</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${apiStats.total_openai_calls || 0}</div>
                    <div class="stat-label">API Calls</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${cachePerf.overall_cache_efficiency?.total_api_calls_saved || 0}</div>
                    <div class="stat-label">Calls Saved</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #6b7280; text-align: center; margin-top: 0.5rem;">
                Cache Size: ${(stats.cache_storage?.total_cache_size_mb || 0).toFixed(1)} MB
            </div>
        `;
    } catch (error) {
        console.error('Error refreshing stats:', error);
        document.getElementById('stats-container').innerHTML = `
            <div style="color: #ef4444; text-align: center;">
                ❌ Error loading stats
            </div>
        `;
    }
}

async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        
        const statusContainer = document.getElementById('system-status');
        const isHealthy = health.status === 'healthy';
        
        statusContainer.innerHTML = `
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span class="status-indicator ${isHealthy ? 'status-online' : 'status-offline'}"></span>
                <strong>${isHealthy ? 'System Online' : 'System Offline'}</strong>
            </div>
            <div style="font-size: 0.9rem; color: #6b7280;">
                <div>RAG System: ${health.rag_system_initialized ? '✅ Ready' : '❌ Not Ready'}</div>
                <div>Uptime: ${new Date(health.timestamp * 1000).toLocaleTimeString()}</div>
            </div>
        `;
    } catch (error) {
        document.getElementById('system-status').innerHTML = `
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-offline"></span>
                <span style="color: #ef4444;">Connection Error</span>
            </div>
        `;
    }
}

async function processDocuments() {
    const statusDiv = document.getElementById('upload-status');
    statusDiv.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            Processing documents...
        </div>
    `;

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ directory: 'uploaded_docs' })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            statusDiv.innerHTML = `
                <div style="color: #10b981; font-weight: 600;">
                    ✅ ${result.message}
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.5rem;">
                    Files processed: ${result.files_processed} | Time: ${result.processing_time?.toFixed(2)}s
                </div>
            `;
            refreshStats(); // Refresh stats after processing
        } else {
            statusDiv.innerHTML = `
                <div style="color: #ef4444; font-weight: 600;">
                    ❌ ${result.detail || 'Processing failed'}
                </div>
            `;
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <div style="color: #ef4444; font-weight: 600;">
                ❌ Network error: ${error.message}
            </div>
        `;
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}
