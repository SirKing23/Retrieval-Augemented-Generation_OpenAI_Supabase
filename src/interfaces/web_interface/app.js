// RAG AI - Document Intelligence Platform JavaScript

class RAGApp {
    constructor() {
        this.currentView = 'dashboard';
        this.sessionId = this.generateSessionId();
        this.notifications = [];
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeTheme();
        this.loadInitialData();

        this.updateSystemStatus();
        
        // Update status every 30 seconds
        setInterval(() => this.updateSystemStatus(), 30000);
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const view = e.currentTarget.dataset.view;
                if (view) this.showView(view);
            });
        });

        // Global search
        document.getElementById('global-search').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.quickSearch();
            }
        });

        // Chat input
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });

        // Document search
        document.getElementById('document-search').addEventListener('input', (e) => {
            this.filterDocuments(e.target.value);
        });

        // Document filter
        document.getElementById('document-filter').addEventListener('change', (e) => {
            this.filterDocuments(document.getElementById('document-search').value, e.target.value);
        });

        // Upload area drag and drop
        const uploadArea = document.getElementById('upload-area');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        });

        // Upload area click to select files (avoid interference with drag/drop)
        uploadArea.addEventListener('click', (e) => {
            // Only trigger file input if the click is directly on the upload area
            // and not on any child elements like buttons
            if (e.target === uploadArea || (e.target.closest('.upload-content') && !e.target.closest('button'))) {
                document.getElementById('file-input').click();
            }
        });

        // Browse button click handler (separate from upload area click)
        const browseBtn = document.getElementById('browse-btn');
        if (browseBtn) {
            browseBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                document.getElementById('file-input').click();
            });
        }

        // File input change
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
            // Clear the input so the same file can be selected again
            e.target.value = '';
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    initializeTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        const themeIcon = document.getElementById('theme-icon');
        if (themeIcon) {
            themeIcon.className = this.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    showView(viewName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });
        document.getElementById(`${viewName}-view`).classList.add('active');

        this.currentView = viewName;

        // Load view-specific data
        this.loadViewData(viewName);
    }

    async loadViewData(viewName) {
        switch (viewName) {
            case 'dashboard':
                await this.loadDashboardData();
                break;
            case 'documents':
                await this.loadDocuments();
                break;
            case 'analytics':
                await this.loadAnalytics();
                break;
        }
    }

    async loadInitialData() {
        await this.loadDashboardData();
        await this.loadDocuments();
    }

    //Dashboard Click
    async loadDashboardData() {
        try {
            const [statsResponse, healthResponse] = await Promise.all([
                fetch('/api/stats'),
                fetch('/api/health')
            ]);

            if (statsResponse.ok) {
                const stats = await statsResponse.json();
                this.updateDashboardCards(stats);
                this.updateRecentActivity();
            }

            if (healthResponse.ok) {
                const health = await healthResponse.json();
                this.updateSystemStatus(health.status === 'healthy');
            }

            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showNotification('Error loading dashboard data', 'error');
        }
    }

    updateDashboardCards(stats) {
        // Update document count
        const docCount = stats.system_health?.cache_storage?.total_files || 0;
        document.getElementById('total-documents').textContent = docCount;


        // Update query count
        const queryCount = stats.api_call_statistics?.total_questions || 0;
        document.getElementById('total-queries').textContent = queryCount;

        // Update response time
        const responseTime = stats.api_call_statistics?.average_response_time || 0;
        document.getElementById('avg-response').textContent = `${Math.round(responseTime * 1000)}ms`;

        // Update cache efficiency
        const cacheHitRate = stats.cache_performance?.hit_rate || 0;
        document.getElementById('cache-efficiency').textContent = `${Math.round(cacheHitRate * 100)}%`;
    }

    updateRecentActivity() {
        const activityList = document.getElementById('recent-activity');
        const activities = [
            { icon: 'fas fa-upload', text: 'New document uploaded', time: '2 minutes ago' },
            { icon: 'fas fa-robot', text: 'AI query processed', time: '5 minutes ago' },
            { icon: 'fas fa-cog', text: 'Cache optimized', time: '10 minutes ago' },
            { icon: 'fas fa-chart-line', text: 'Analytics updated', time: '15 minutes ago' }
        ];

        activityList.innerHTML = activities.map(activity => `
            <div class="activity-item">
                <div class="activity-icon">
                    <i class="${activity.icon}"></i>
                </div>
                <div class="activity-content">
                    <div class="activity-text">${activity.text}</div>
                    <div class="activity-time">${activity.time}</div>
                </div>
            </div>
        `).join('');
    }

    async loadDocuments() {
        try {
            this.showLoading(true);
            const response = await fetch('/api/files');
            
            if (response.ok) {
                const files = await response.json();
                this.displayDocuments(files);
            } else {
                throw new Error('Failed to load documents');
            }
        } catch (error) {
            console.error('Error loading documents:', error);
            this.showNotification('Error loading documents', 'error');
            this.displayDocuments([]);
        } finally {
            this.showLoading(false);
        }
    }

    displayDocuments(files) {
        const container = document.getElementById('documents-list');
               
        if (files.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-file-alt"></i>
                    <h3>No documents found</h3>
                    <p>Upload some documents to get started</p>
                    <button class="primary-btn" onclick="ragApp.showView('upload')">
                        <i class="fas fa-plus"></i>
                        Upload Documents
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = files.map(file => {
            const fileSize = this.formatFileSize(file.size);
            const fileType = file.type.toLowerCase();
            
            return `
                <div class="document-item" data-file-id="${file.id}">
                    <div class="document-icon ${fileType}">
                        <i class="fas fa-file-${this.getFileIcon(fileType)}"></i>
                    </div>
                    <div class="document-info">
                        <div class="document-name">${file.name}</div>
                        <div class="document-meta">${fileSize} • ${file.uploaded}</div>
                    </div>
                    <div class="document-actions">
                        <button class="document-btn" onclick="ragApp.viewDocument('${file.id}')" title="View">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="document-btn" onclick="ragApp.queryDocument('${file.id}')" title="Query">
                            <i class="fas fa-search"></i>
                        </button>
                        <button class="document-btn" onclick="ragApp.deleteDocument('${file.id}')" title="Delete">
                            <i class="fas fa-trash"></i>
                     
                    </div>
                </div>
            `;
        }).join('');
    }

    getFileIcon(fileType) {
        const icons = {
            'pdf': 'pdf',
            'txt': 'alt',
            'docx': 'word',
            'doc': 'word'
        };
        return icons[fileType] || 'alt';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    filterDocuments(searchTerm, fileType = 'all') {
        const items = document.querySelectorAll('.document-item');
        
        items.forEach(item => {
            const name = item.querySelector('.document-name').textContent.toLowerCase();
            const type = item.dataset.fileType || '';
            
            const matchesSearch = searchTerm === '' || name.includes(searchTerm.toLowerCase());
            const matchesType = fileType === 'all' || type === fileType;
            
            item.style.display = matchesSearch && matchesType ? 'flex' : 'none';
        });
    }

    async quickSearch() {
        const query = document.getElementById('global-search').value.trim();
        if (!query) return;

        // Switch to chat view and send message
        this.showView('chat');
        document.getElementById('chat-input').value = query;
        await this.sendChatMessage();
        
        // Clear global search
        document.getElementById('global-search').value = '';
    }

    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Clear input
        input.value = '';

        // Add user message to chat
        this.addChatMessage(message, 'user');

        try {
            // Show typing indicator
            this.showTypingIndicator();

            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: message,
                    session_id: this.sessionId
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.removeTypingIndicator();
                this.addChatMessage(data.response, 'assistant', data.sources);
            } else {
                throw new Error('Failed to get response');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.removeTypingIndicator();
            this.addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
            this.showNotification('Error sending message', 'error');
        }
    }

    addChatMessage(message, sender, sources = []) {
        const messagesContainer = document.getElementById('chat-messages');
        const timestamp = new Date().toLocaleTimeString();
        
        const messageHTML = `
            <div class="message-group ${sender}">
                <div class="message-avatar">
                    <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">${this.formatMessage(message)}</div>
                    ${sources && sources.length > 0 ? this.formatSources(sources) : ''}
                    <div class="message-time">${timestamp}</div>
                </div>
            </div>
        `;
        
        messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    formatMessage(message) {
        // First escape HTML characters to prevent XSS and preserve user input
        const escapedMessage = this.escapeHtml(message);
        
        // Then apply basic markdown-like formatting to the escaped content
        return escapedMessage
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatSources(sources) {
        if (!sources || sources.length === 0) return '';
        
        return `
            <div class="message-sources">
                <div class="sources-title">Sources:</div>
                ${sources.map(source => {
                    const filename = source.filename || source.title || source.source_id || 'Unknown Document';
                    const pageInfo = source.page_number ? ` (Page ${source.page_number})` : 
                                   source.chunk_index > 0 ? ` (Section ${source.chunk_index + 1})` : '';
                    const fileType = source.file_type ? source.file_type.toLowerCase() : 'file';
                    const icon = fileType === 'pdf' ? 'fa-file-pdf' : 
                               fileType === 'docx' || fileType === 'doc' ? 'fa-file-word' : 
                               'fa-file-alt';
                    
                    // Generate document URL using the new /documents mount point
                    let fileUrl = '';
                    if (source.filename) {
                        fileUrl = `/documents/${encodeURIComponent(source.filename)}`;
                    }
                    
                    return `
                        <div class="source-item">
                            <i class="fas ${icon}"></i>
                            ${fileUrl ? `<a href="${fileUrl}" target="_blank" rel="noopener">${filename}${pageInfo}</a>` : `<span>${filename}${pageInfo}</span>`}
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingHTML = `
            <div class="message-group assistant typing-indicator">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        
        messagesContainer.insertAdjacentHTML('beforeend', typingHTML);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    handleFiles(files) {
        const fileArray = Array.from(files);
        const validFiles = fileArray.filter(file => this.isValidFile(file));
        
        if (validFiles.length === 0) {
            this.showNotification('No valid files selected', 'warning');
            return;
        }

        this.displayUploadQueue(validFiles);
    }

    isValidFile(file) {
        const validTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        const validExtensions = ['.pdf', '.txt', '.docx', '.doc'];
        
        return validTypes.includes(file.type) || 
               validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    }

    displayUploadQueue(files) {
        const queueContainer = document.getElementById('upload-queue');
        
        queueContainer.innerHTML = `
            <div class="upload-queue-header">
                <h4>Upload Queue (${files.length} files)</h4>
                <button class="primary-btn" id="upload-all-btn">
                    <i class="fas fa-upload"></i>
                    Upload All
                </button>
            </div>
            <div class="upload-items">
                ${files.map((file, index) => `
                    <div class="upload-item" data-index="${index}">
                        <div class="upload-file-info">
                            <i class="fas fa-file-${this.getFileIcon(file.type)}"></i>
                            <div class="upload-file-details">
                                <div class="upload-file-name">${file.name}</div>
                                <div class="upload-file-size">${this.formatFileSize(file.size)}</div>
                            </div>
                        </div>
                        <div class="upload-progress">
                            <div class="progress-bar">
                                <div class="progress-fill"></div>
                            </div>
                            <div class="upload-status">Ready</div>
                        </div>
                        <button class="upload-remove" data-remove-index="${index}">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `).join('')}
            </div>
        `;
        
        this.uploadQueue = files;
        
        // Add event listeners after creating the HTML
        const uploadAllBtn = document.getElementById('upload-all-btn');
        if (uploadAllBtn) {
            uploadAllBtn.addEventListener('click', () => this.startUpload());
        }
        
        // Add remove button event listeners
        const removeButtons = queueContainer.querySelectorAll('.upload-remove');
        removeButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const index = parseInt(e.currentTarget.getAttribute('data-remove-index'));
                this.removeFromQueue(index);
            });
        });
    }

    removeFromQueue(index) {
        this.uploadQueue.splice(index, 1);
        this.displayUploadQueue(this.uploadQueue);
    }

    async startUpload() {
        if (!this.uploadQueue || this.uploadQueue.length === 0) return;

        try {
            for (let i = 0; i < this.uploadQueue.length; i++) {
                await this.uploadFile(i);
            }

            this.showNotification('Files uploaded successfully!', 'success');
            this.uploadQueue = [];
            document.getElementById('upload-queue').innerHTML = '';
            
            // Refresh documents list
            await this.loadDocuments();
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification('Error uploading files', 'error');
        }
    }

    async uploadFile(index) {
        const file = this.uploadQueue[index];
        const item = document.querySelector(`[data-index="${index}"]`);
        const progressFill = item.querySelector('.progress-fill');
        const status = item.querySelector('.upload-status');
        
        status.textContent = 'Uploading...';
        progressFill.style.width = '0%';

        try {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();
            
            // Track upload progress
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = (e.loaded / e.total) * 100;
                    progressFill.style.width = `${progress}%`;
                }
            });

            // Handle upload completion
            const uploadPromise = new Promise((resolve, reject) => {
                xhr.onload = () => {
                    if (xhr.status === 200) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        reject(new Error(`Upload failed: ${xhr.status}`));
                    }
                };
                xhr.onerror = () => reject(new Error('Upload failed'));
            });

            xhr.open('POST', '/api/upload-file', true);
            xhr.send(formData);

            await uploadPromise;
            
            progressFill.style.width = '100%';
            status.textContent = 'Complete';
            status.style.color = 'var(--success-color)';
            
        } catch (error) {
            status.textContent = 'Failed';
            status.style.color = 'var(--error-color)';
            throw error;
        }
    }

    async simulateFileUpload(index) {
        const item = document.querySelector(`[data-index="${index}"]`);
        const progressFill = item.querySelector('.progress-fill');
        const status = item.querySelector('.upload-status');
        
        status.textContent = 'Uploading...';
        
        // Simulate progress
        for (let progress = 0; progress <= 100; progress += 10) {
            progressFill.style.width = `${progress}%`;
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        status.textContent = 'Complete';
        status.style.color = 'var(--success-color)';
    }

    async startProcessing() {
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    directory: './data/Knowledge_Base_Files'
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(`Successfully processed ${result.files_processed} files`, 'success');
                await this.loadDocuments();
            } else {
                throw new Error('Processing failed');
            }
        } catch (error) {
            console.error('Error processing documents:', error);
            this.showNotification('Error processing documents', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async loadAnalytics() {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const stats = await response.json();
                this.updateAnalyticsCards(stats);
            }
        } catch (error) {
            console.error('Error loading analytics:', error);
        }
    }

    updateAnalyticsCards(stats) {
        // Cache Hit Rate
        const cacheHitRate = stats.cache_performance?.hit_rate || 0;
        document.getElementById('cache-hit-rate').textContent = `${Math.round(cacheHitRate * 100)}%`;

        // API Calls
        const apiCalls = stats.api_call_statistics?.total_api_calls || 0;
        document.getElementById('api-calls').textContent = apiCalls;

        // Storage Used
        const storageUsed = stats.system_health?.cache_storage?.total_size_mb || 0;
        document.getElementById('storage-used').textContent = `${Math.round(storageUsed)} MB`;

        // Response Time
        const responseTime = stats.api_call_statistics?.average_response_time || 0;
        document.getElementById('response-time').textContent = `${Math.round(responseTime * 1000)}ms`;
    }

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', this.theme);
        this.initializeTheme();
    }

    showNotifications() {
        // Toggle notification panel (placeholder)
        this.showNotification('No new notifications', 'info');
    }

    updateSystemStatus(isOnline = true) {
        const statusIndicator = document.getElementById('system-status');
        const statusText = document.getElementById('status-text');
        
        if (statusIndicator && statusText) {
            statusIndicator.className = `status-indicator ${isOnline ? 'online' : 'offline'}`;
            statusText.textContent = isOnline ? 'Online' : 'Offline';
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }

    showNotification(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastIcon = toast.querySelector('.toast-icon i');
        const toastMessage = toast.querySelector('.toast-message');
        
        // Set icon based on type
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        toastIcon.className = icons[type] || icons.info;
        toastMessage.textContent = message;
        
        // Show toast
        toast.classList.add('show');
        
        // Hide after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // Suggestion handlers
    useSuggestion(button) {
        const suggestion = button.textContent;
        document.getElementById('chat-input').value = suggestion;
        this.sendChatMessage();
    }

    // Document actions
    viewDocument(fileId) {
        this.showNotification(`Viewing document: ${fileId}`, 'info');
        // Implement document viewer
    }

    queryDocument(fileId) {
        this.showView('chat');
        document.getElementById('chat-input').value = `Tell me about the document with ID: ${fileId}`;
        this.sendChatMessage();
    }

    async deleteDocument(fileId) {
        if (!confirm('Are you sure you want to delete this document?')) return;

        try {
            // Delete the file from DOCUMENTS_DIR via API
            const fileDeleteResponse = await fetch(`/api/delete-file/${encodeURIComponent(fileId)}`, {
                method: 'DELETE'
            });

            if (fileDeleteResponse.ok) {
                this.showNotification(`Document ${fileId} deleted`, 'success');
                this.loadDocuments();

                // Ask if user wants to delete vector embeddings
                if (confirm('Do you also want to delete the vector embeddings associated with this document?')) {
                    const embeddingDeleteResponse = await fetch(`/api/delete-embeddings/${encodeURIComponent(fileId)}`, {
                        method: 'DELETE'
                    });

                    if (embeddingDeleteResponse.ok) {
                        this.showNotification('Vector embeddings deleted successfully', 'success');
                    } else {
                        this.showNotification('Failed to delete vector embeddings', 'error');
                    }
                }
            } else {
                this.showNotification('Failed to delete document', 'error');
            }
        } catch (error) {
            console.error('Error deleting document:', error);
            this.showNotification('Error deleting document', 'error');
        }
    }

    // Chat actions
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            document.getElementById('chat-messages').innerHTML = `
                <div class="message-group assistant">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-text">
                            Hello! I'm your AI document assistant. I can help you find information from your uploaded documents. What would you like to know?
                        </div>
                        <div class="message-time">Just now</div>
                    </div>
                </div>
            `;
            this.showNotification('Chat cleared', 'success');
        }
    }

    exportChat() {
        const messages = document.querySelectorAll('.message-group');
        let chatContent = 'RAG AI Chat Export\n' + '='.repeat(50) + '\n\n';
        
        messages.forEach(msg => {
            const sender = msg.classList.contains('user') ? 'User' : 'Assistant';
            const text = msg.querySelector('.message-text').textContent;
            const time = msg.querySelector('.message-time').textContent;
            
            chatContent += `[${time}] ${sender}: ${text}\n\n`;
        });
        
        const blob = new Blob([chatContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag_chat_${new Date().toISOString().slice(0, 10)}.txt`;
        a.click();
        
        this.showNotification('Chat exported successfully', 'success');
    }

    // Document view toggles
    toggleDocumentView(viewType) {
        const buttons = document.querySelectorAll('.control-btn');
        buttons.forEach(btn => btn.classList.remove('active'));
        
        event.target.classList.add('active');
        
        // Implement grid vs list view
        this.showNotification(`Switched to ${viewType} view`, 'info');
    }
}

// Global functions for onclick handlers
function quickSearch() {
    ragApp.quickSearch();
}

function sendChatMessage() {
    ragApp.sendChatMessage();
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        ragApp.sendChatMessage();
    }
}

function toggleTheme() {
    ragApp.toggleTheme();
}

function showNotifications() {
    ragApp.showNotifications();
}

function clearChat() {
    ragApp.clearChat();
}

function exportChat() {
    ragApp.exportChat();
}

function useSuggestion(button) {
    ragApp.useSuggestion(button);
}

function toggleDocumentView(viewType) {
    ragApp.toggleDocumentView(viewType);
}

function showView(viewName) {
    ragApp.showView(viewName);
}

function startProcessing() {
    ragApp.startProcessing();
}

function startUpload() {
    if (ragApp && ragApp.startUpload) {
        ragApp.startUpload();
    }
}

// Initialize the app when the page loads
let ragApp;
document.addEventListener('DOMContentLoaded', () => {
    ragApp = new RAGApp();
});

// Add some additional CSS for features not in the main CSS
const additionalStyles = `
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: var(--spacing-2xl);
        color: var(--text-muted);
    }
    
    .empty-state i {
        font-size: 64px;
        margin-bottom: var(--spacing-lg);
        color: var(--text-muted);
    }
    
    .empty-state h3 {
        font-size: 24px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-md);
    }
    
    .empty-state p {
        margin-bottom: var(--spacing-xl);
    }
    
    /* Activity Items */
    .activity-item {
        display: flex;
        align-items: center;
        padding: var(--spacing-md);
        border-bottom: 1px solid var(--border-light);
        gap: var(--spacing-md);
    }
    
    .activity-item:last-child {
        border-bottom: none;
    }
    
    .activity-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--primary-light);
        color: var(--primary-color);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    
    .activity-content {
        flex: 1;
    }
    
    .activity-text {
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
    }
    
    .activity-time {
        font-size: 12px;
        color: var(--text-muted);
    }
    
    /* Message Sources */
    .message-sources {
        margin-top: var(--spacing-md);
        padding-top: var(--spacing-md);
        border-top: 1px solid var(--border-light);
    }
    
    .sources-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: var(--spacing-sm);
    }
    
    .source-item {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        font-size: 12px;
        color: var(--text-secondary);
        margin-bottom: var(--spacing-xs);
    }
    
    /* Typing Dots */
    .typing-dots {
        display: flex;
        gap: 4px;
        padding: var(--spacing-md);
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--text-muted);
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Upload Queue */
    .upload-queue-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--spacing-lg);
        padding-bottom: var(--spacing-md);
        border-bottom: 1px solid var(--border-color);
    }
    
    .upload-queue-header h4 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .upload-item {
        display: flex;
        align-items: center;
        padding: var(--spacing-md);
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        margin-bottom: var(--spacing-sm);
        gap: var(--spacing-md);
    }
    
    .upload-file-info {
        display: flex;
        align-items: center;
        gap: var(--spacing-md);
        flex: 1;
    }
    
    .upload-file-info i {
        font-size: 24px;
        color: var(--primary-color);
    }
    
    .upload-file-name {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    .upload-file-size {
        font-size: 12px;
        color: var(--text-muted);
    }
    
    .upload-progress {
        flex: 1;
        margin: 0 var(--spacing-md);
    }
    
    .progress-bar {
        width: 100%;
        height: 4px;
        background: var(--border-color);
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: var(--spacing-xs);
    }
    
    .progress-fill {
        height: 100%;
        background: var(--primary-color);
        width: 0%;
        transition: width 0.3s ease;
    }
    
    .upload-status {
        font-size: 12px;
        color: var(--text-muted);
    }
    
    .upload-remove {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        padding: var(--spacing-xs);
        border-radius: var(--border-radius);
        transition: var(--transition);
    }
    
    .upload-remove:hover {
        background: var(--error-color);
        color: white;
    }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);
