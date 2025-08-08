# Method Interaction Diagram

## Overview
This diagram illustrates the interaction between the frontend (`app.js`), backend (`rag_webapp.py`), and core logic (`RAG_Core.py`) in the Retrieval-Augmented Generation (RAG) system.

```mermaid
graph TD
    subgraph Frontend [Frontend: app.js]
        A[sendChatMessage] -->|API Call| B[ask_question]
        C[startUpload] -->|API Call| D[upload_documents]
        E[toggleDocumentView] -->|UI Update| F[Backend Response]
    end

    subgraph Backend [Backend: rag_webapp.py]
        B[ask_question] -->|Calls| G[answer_this]
        D[upload_documents] -->|Calls| H[initialize_files]
        I[get_files] -->|Fetches| J[File Data]
        K[clear_cache] -->|Calls| L[Cache Manager]
    end

    subgraph Core [Core Logic: RAG_Core.py]
        G[answer_this] -->|Uses| M[generate_embedding]
        G -->|Uses| N[_manage_context_window]
        G -->|Uses| O[_extract_source_info]
        H[initialize_files] -->|Processes| P[File Processing]
        L[Cache Manager] -->|Manages| Q[Cache Data]
    end

    Frontend --> Backend
    Backend --> Core
```

## Explanation
1. **Frontend (`app.js`)**:
   - Handles user interactions and sends API requests to the backend.
   - Updates the UI based on responses from the backend.

2. **Backend (`rag_webapp.py`)**:
   - Exposes API endpoints for the frontend.
   - Interacts with the core logic in `RAG_Core.py` to process requests.

3. **Core Logic (`RAG_Core.py`)**:
   - Implements the main functionality of the RAG system, including embedding generation, context management, and file processing.

## Key Interactions
- `sendChatMessage` in `app.js` calls `ask_question` in `rag_webapp.py`, which invokes `answer_this` in `RAG_Core.py`.
- `startUpload` in `app.js` calls `upload_documents` in `rag_webapp.py`, which processes files using `initialize_files` in `RAG_Core.py`.
- Cache management and context optimization are handled internally within `RAG_Core.py`.
