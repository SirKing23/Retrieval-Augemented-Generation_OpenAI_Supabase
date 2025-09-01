# Document Relevance Assessment Enhancement

## Overview
This enhancement adds intelligent document relevance assessment to the RAG system to prevent irrelevant sources from being included in responses when the retrieved documents don't actually help answer the user's query.

## Problem Solved
Previously, when a user asked a general knowledge question like "which postgres database is better? weaviate or supabase?", the system would:
1. Route the query to the knowledge base (correct behavior)
2. Retrieve documents from the knowledge base based on keyword similarity
3. Include those documents as sources even if they weren't relevant to the specific question
4. Provide misleading source citations

## Solution Implemented

### 1. Document Relevance Assessment Method
- **Function**: `assess_document_relevance()`
- **Purpose**: Uses LLM to evaluate if retrieved documents are actually relevant to the user's question
- **Categories**: 
  - `HIGHLY_RELEVANT`: Documents directly address the question
  - `SOMEWHAT_RELEVANT`: Documents contain related information  
  - `NOT_RELEVANT`: Documents don't contain relevant information

### 2. Enhanced Answer Logic
The `answer_this()` method now:
1. Retrieves documents from vector database as before
2. **NEW**: Assesses document relevance using LLM
3. **NEW**: Decides whether to include sources based on relevance assessment
4. **NEW**: Falls back to direct answer for irrelevant documents + general knowledge queries
5. Provides appropriate response with or without source citations

### 3. Intelligent Source Inclusion
- **Relevant documents**: Sources are included and cited
- **Irrelevant documents**: Sources are excluded, general knowledge response provided
- **Mixed relevance**: Decision based on query classification (domain-specific vs general)

### 4. Enhanced System Prompts
- When documents are relevant: Instructs LLM to cite sources
- When documents are irrelevant: Instructs LLM to use general knowledge and not cite irrelevant sources

## Configuration
The relevance assessment uses configurable parameters from `.env`:
- `CLASSIFICATION_MODEL`: Model for relevance assessment (default: gpt-3.5-turbo)
- `CLASSIFICATION_MAX_TOKENS`: Max tokens for assessment (default: 20)
- `CLASSIFICATION_TEMPERATURE`: Temperature for assessment (default: 0.1)

## Statistics Tracking
New statistic added:
- `sources_excluded_irrelevant`: Number of times sources were excluded due to irrelevance

## Example Behavior

### Before Enhancement:
**Query**: "which postgres database is better? weaviate or supabase?"
**Result**: Returns answer with irrelevant sources from knowledge base documents

### After Enhancement:
**Query**: "which postgres database is better? weaviate or supabase?"
**Process**:
1. Retrieves documents from knowledge base
2. Assesses documents as not relevant to database comparison question
3. Provides general knowledge answer about PostgreSQL databases
4. **No irrelevant sources included**

## Benefits
1. **Accurate Source Attribution**: Only relevant documents are cited as sources
2. **Better User Experience**: Users get appropriate answers without misleading citations  
3. **Flexible Routing**: Can handle both domain-specific and general knowledge queries appropriately
4. **Transparency**: System tracks when and why sources are excluded
5. **Configurable**: Assessment parameters can be tuned via environment variables
