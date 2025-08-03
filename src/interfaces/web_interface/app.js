let sessionId = 'session_' + Math.random().toString(36).substr(2, 12);
let messageCounter = 0;
let currentSection = 'files';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    refreshStats();
    loadFiles();
    checkSystemHealth();
    setInterval(refreshStats, 30000); // Auto-refresh every 30 seconds
    setInterval(checkSystemHealth, 60000); // Check health every minute
    
    // Initialize file upload handler
    const fileUpload = document.getElementById('file-upload');
    if (fileUpload) {
        fileUpload.addEventListener('change', handleFileUpload);
    }
});

// Section Navigation
function showSection(sectionName) {
    // Update navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[onclick="showSection('${sectionName}')"]`).classList.add('active');
    
    // Update content sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(`${sectionName}-section`).classList.add('active');
    
    currentSection = sectionName;
    
    // Load section-specific data
    switch(sectionName) {
        case 'files':
            loadFiles();
            break;
        case 'trash':
            loadTrash();
            break;
        case 'statistics':
            refreshStats();
            break;
    }
}

// File Upload Functions
function openFileUpload() {
    document.getElementById('file-upload').click();
}

async function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            showNotification('Files uploaded successfully!', 'success');
            loadFiles(); // Refresh file list
        } else {
            showNotification('Upload failed!', 'error');
        }
    } catch (error) {
        showNotification('Upload error: ' + error.message, 'error');
    }
}

async function loadFiles() {
    const container = document.getElementById('files-list');
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading files...</div>';
    
    try {
        const response = await fetch('/api/files');
        const files = await response.json();
        
        if (files.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📁</div>
                    <div class="empty-text">No files uploaded yet</div>
                </div>
            `;
            return;
        }
        
        container.innerHTML = files.map(file => `
            <div class="file-card" onclick="selectFile('${file.id}')">
                <div class="file-icon">${getFileIcon(file.type)}</div>
                <div class="file-name">${file.name}</div>
                <div class="file-details">
                    <div>${formatFileSize(file.size)}</div>
                    <div>${formatDate(file.uploaded)}</div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">❌</div>
                <div class="empty-text">Error loading files</div>
            </div>
        `;
    }
}

async function loadTrash() {
    const container = document.getElementById('trash-list');
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading deleted files...</div>';
    
    try {
        const response = await fetch('/api/trash');
        const files = await response.json();
        
        if (files.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🗑️</div>
                    <div class="empty-text">No deleted files</div>
                </div>
            `;
            return;
        }
        
        container.innerHTML = files.map(file => `
            <div class="file-card">
                <div class="file-icon">${getFileIcon(file.type)}</div>
                <div class="file-name">${file.name}</div>
                <div class="file-details">
                    <div>${formatFileSize(file.size)}</div>
                    <div>Deleted: ${formatDate(file.deleted)}</div>
                </div>
                <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
                    <button class="btn btn-secondary btn-sm" onclick="restoreFile('${file.id}')">
                        🔄 Restore
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="permanentDelete('${file.id}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">❌</div>
                <div class="empty-text">Error loading trash</div>
            </div>
        `;
    }
}

// Utility Functions
function getFileIcon(fileType) {
    const icons = {
        'pdf': '📄',
        'txt': '📝',
        'docx': '📄',
        'doc': '📄',
        'default': '📄'
    };
    return icons[fileType] || icons.default;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString();
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Chat Functions
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
        
        // Update statistics in the statistics section
        const apiCallsElem = document.getElementById('api-calls');
        const cacheHitsElem = document.getElementById('cache-hits');
        const documentsElem = document.getElementById('documents-processed');
        const responseTimeElem = document.getElementById('avg-response-time');
        
        if (apiCallsElem) apiCallsElem.textContent = stats.api_call_statistics?.total_openai_calls || 0;
        if (cacheHitsElem) cacheHitsElem.textContent = stats.cache_performance?.overall_cache_efficiency?.total_api_calls_saved || 0;
        if (documentsElem) documentsElem.textContent = stats.documents_processed || 0;
        if (responseTimeElem) responseTimeElem.textContent = stats.avg_response_time ? `${stats.avg_response_time.toFixed(0)}ms` : '0ms';
        
        // Update storage information
        if (stats.cache_storage) {
            const usedMB = stats.cache_storage.total_cache_size_mb || 0;
            const totalMB = 1000; // 1GB limit
            const usedPercentage = (usedMB / totalMB * 100).toFixed(1);
            
            const storageUsedBar = document.querySelector('.storage-used');
            const storageUsedText = document.getElementById('storage-used');
            const storageTotalText = document.getElementById('storage-total');
            
            if (storageUsedBar) storageUsedBar.style.width = `${usedPercentage}%`;
            if (storageUsedText) storageUsedText.textContent = `${usedMB.toFixed(1)} MB`;
            if (storageTotalText) storageTotalText.textContent = `${totalMB} MB`;
        }

        // Legacy support for the old stats container if it exists
        const container = document.getElementById('stats-container');
        if (container) {
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
        }
    } catch (error) {
        console.error('Error refreshing stats:', error);
        const container = document.getElementById('stats-container');
        if (container) {
            container.innerHTML = `
                <div style="color: #ef4444; text-align: center;">
                    ❌ Error loading stats
                </div>
            `;
        }
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

// Additional functions for the new interface

function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="message assistant">
            <div class="message-header">🤖 Assistant</div>
            <div class="message-content">
                Welcome! I can help you find information from your documents. Ask me anything about the uploaded documents.
            </div>
        </div>
    `;
    messageCounter = 0;
}

// File Management Functions
async function selectFile(fileId) {
    console.log('Selected file:', fileId);
    // Implement file selection logic - could highlight file or show preview
}

async function deleteFile(fileId) {
    try {
        const response = await fetch(`/api/files/${fileId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('File moved to trash', 'success');
            loadFiles();
        } else {
            showNotification('Failed to delete file', 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

async function restoreFile(fileId) {
    try {
        const response = await fetch(`/api/trash/${fileId}/restore`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showNotification('File restored successfully', 'success');
            loadTrash();
            if (currentSection === 'files') loadFiles();
        } else {
            showNotification('Failed to restore file', 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

async function permanentDelete(fileId) {
    if (!confirm('Are you sure you want to permanently delete this file?')) return;
    
    try {
        const response = await fetch(`/api/trash/${fileId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('File permanently deleted', 'success');
            loadTrash();
        } else {
            showNotification('Failed to delete file', 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

async function emptyTrash() {
    if (!confirm('Are you sure you want to empty the trash? This cannot be undone.')) return;
    
    try {
        const response = await fetch('/api/trash', {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('Trash emptied successfully', 'success');
            loadTrash();
        } else {
            showNotification('Failed to empty trash', 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

async function refreshFiles() {
    loadFiles();
}
