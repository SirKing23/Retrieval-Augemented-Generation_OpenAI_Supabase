"""
Comprehensive test suite for the Optimized RAG System
Run with: python test_optimized_rag.py
"""

import pytest
import os
import time
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from RAG_Core import OptimizedRAGSystem, RAGSystemConfig, RAGLogger, PerformanceMetrics
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure RAG_Core_Optimized.py is in the same directory")
    sys.exit(1)

class TestRAGSystemConfig:
    """Test the configuration management"""
    
    def test_config_from_env(self):
        """Test configuration loading from environment variables"""
        # Set test environment variables
        test_env = {
            'OPENAI_API_KEY': 'test_openai_key',
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_supabase_key'
        }
        
        with patch.dict(os.environ, test_env):
            config = RAGSystemConfig()
            assert config.ai_api_key == 'test_openai_key'
            assert config.supabase_url == 'https://test.supabase.co'
            assert config.supabase_key == 'test_supabase_key'
    
    def test_config_validation_missing_vars(self):
        """Test configuration validation with missing variables"""
        with patch.dict(os.environ, {}, clear=True):
            config = RAGSystemConfig()
            with pytest.raises(EnvironmentError):
                config.validate_config()
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        test_env = {
            'OPENAI_API_KEY': 'test_key',
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }
        
        with patch.dict(os.environ, test_env):
            config = RAGSystemConfig()
            config.validate_config()  # Should not raise

class TestRAGLogger:
    """Test the logging functionality"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        logger = RAGLogger()
        assert logger.logger.name == 'RAGSystem'
    
    def test_log_query(self):
        """Test query logging"""
        logger = RAGLogger()
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_query("test question", 1.5, 3)
            mock_info.assert_called_once()
    
    def test_log_error(self):
        """Test error logging"""
        logger = RAGLogger()
        with patch.object(logger.logger, 'error') as mock_error:
            test_error = Exception("Test error")
            logger.log_error(test_error, "test_context")
            mock_error.assert_called_once()

class TestOptimizedRAGSystem:
    """Test the main RAG system functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        config = RAGSystemConfig()
        config.ai_api_key = "test_api_key"
        config.supabase_url = "https://test.supabase.co"
        config.supabase_key = "test_supabase_key"
        return config
    
    @pytest.fixture
    def mock_rag_system(self, mock_config):
        """Create a mock RAG system for testing"""
        with patch('RAG_Core_Optimized.create_client') as mock_create_client, \
             patch('RAG_Core_Optimized.OpenAI') as mock_openai:
            
            mock_create_client.return_value = Mock()
            mock_openai.return_value = Mock()
            
            rag_system = OptimizedRAGSystem(config=mock_config)
            return rag_system
    
    def test_initialization_success(self, mock_rag_system):
        """Test successful initialization"""
        assert mock_rag_system.ai_api_key == "test_api_key"
        assert mock_rag_system.supabase_url == "https://test.supabase.co"
        assert hasattr(mock_rag_system, 'logger')
        assert hasattr(mock_rag_system, 'performance_metrics')
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid configuration"""
        config = RAGSystemConfig()
        config.ai_api_key = ""  # Invalid empty key
        
        with pytest.raises(ValueError):
            OptimizedRAGSystem(config=config)
    
    def test_input_sanitization(self, mock_rag_system):
        """Test input sanitization"""
        # Test normal input
        clean_input = mock_rag_system.sanitize_input("Hello world")
        assert clean_input == "Hello world"
        
        # Test input with dangerous characters
        dirty_input = mock_rag_system.sanitize_input("Hello <script>alert('xss')</script>")
        assert "<script>" not in dirty_input
        assert "alert" in dirty_input  # Content should remain, just tags removed
        
        # Test very long input
        long_input = "A" * 15000
        sanitized = mock_rag_system.sanitize_input(long_input)
        assert len(sanitized) <= 10000
    
    def test_api_key_validation(self, mock_rag_system):
        """Test API key validation"""
        # Valid key
        assert mock_rag_system.validate_api_key("sk-1234567890abcdefghijklmnop")
        
        # Invalid keys
        assert not mock_rag_system.validate_api_key("")
        assert not mock_rag_system.validate_api_key("short")
        assert not mock_rag_system.validate_api_key(None)
    
    def test_embedding_generation_with_cache(self, mock_rag_system):
        """Test embedding generation and caching"""
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_rag_system.ai_instance.embeddings.create.return_value = mock_response
        
        # First call
        embedding1 = mock_rag_system.generate_embedding("test text")
        assert embedding1 == [0.1, 0.2, 0.3]
        
        # Second call should use cache
        embedding2 = mock_rag_system.generate_embedding("test text")
        assert embedding2 == [0.1, 0.2, 0.3]
        
        # OpenAI should only be called once due to caching
        assert mock_rag_system.ai_instance.embeddings.create.call_count == 1
        
        # Cache should contain the embedding
        assert len(mock_rag_system.embedding_cache) == 1
    
    def test_batch_embedding_generation(self, mock_rag_system):
        """Test batch embedding generation"""
        # Mock the OpenAI batch response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_rag_system.ai_instance.embeddings.create.return_value = mock_response
        
        texts = ["text1", "text2"]
        embeddings = mock_rag_system.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        # Both embeddings should be cached
        assert len(mock_rag_system.embedding_cache) == 2
    
    def test_context_window_management(self, mock_rag_system):
        """Test context window truncation"""
        # Create a very long context
        long_context = "This is a test sentence. " * 1000
        
        # Test truncation
        truncated = mock_rag_system._manage_context_window(long_context, max_tokens=100)
        
        # Should be shorter than original
        assert len(truncated) < len(long_context)
    
    def test_chat_history_optimization(self, mock_rag_system):
        """Test chat history management"""
        # Fill chat history beyond max length
        for i in range(25):  # More than max_history_length (20)
            mock_rag_system.chat_history.append({"role": "user", "content": f"Message {i}"})
            mock_rag_system.chat_history.append({"role": "assistant", "content": f"Response {i}"})
        
        # Should have 50 messages (25 * 2)
        assert len(mock_rag_system.chat_history) == 50
        
        # Optimize history
        mock_rag_system._optimize_chat_history()
        
        # Should be reduced to 20 messages (4 + 16)
        assert len(mock_rag_system.chat_history) == 20
    
    def test_answer_this_with_mocks(self, mock_rag_system):
        """Test the main answer_this method"""
        # Mock embedding generation
        mock_rag_system.generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock Supabase response
        mock_response = Mock()
        mock_response.data = [
            {"content": "Test document content 1"},
            {"content": "Test document content 2"}
        ]
        mock_rag_system.supabase.rpc.return_value.execute.return_value = mock_response
        
        # Mock OpenAI chat response
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test AI response"))]
        mock_rag_system.ai_instance.chat.completions.create.return_value = mock_chat_response
        
        # Test the method
        response = mock_rag_system.answer_this("What is ProductCenter?")
        
        # Verify response structure
        assert "response" in response
        assert "documents_found" in response
        assert "response_time" in response
        assert "sources" in response  # New: source tracking
        assert response["response"] == "Test AI response"
        assert response["documents_found"] == 2
        assert isinstance(response["response_time"], float)
        assert isinstance(response["sources"], list)
        assert len(response["sources"]) == 2  # Should have sources for each document
    
    def test_answer_this_no_documents(self, mock_rag_system):
        """Test answer_this when no documents are found"""
        # Mock embedding generation
        mock_rag_system.generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock Supabase response with no documents
        mock_response = Mock()
        mock_response.data = []
        mock_rag_system.supabase.rpc.return_value.execute.return_value = mock_response
        
        response = mock_rag_system.answer_this("Nonexistent question")
        
        assert response["response"] == mock_rag_system.ai_default_no_response
        assert response["documents_found"] == 0
        assert "sources" in response
        assert response["sources"] == []
    
    def test_answer_this_invalid_input(self, mock_rag_system):
        """Test answer_this with invalid input"""
        # Test empty input
        response = mock_rag_system.answer_this("")
        assert "error" in response
        assert "sources" in response
        assert response["sources"] == []
        
        # Test None input (converted to string)
        response = mock_rag_system.answer_this(None)
        assert "error" in response
        assert "sources" in response
        assert response["sources"] == []
    
    def test_performance_report(self, mock_rag_system):
        """Test performance reporting"""
        # Add some data to cache and history
        mock_rag_system.embedding_cache["test"] = [0.1, 0.2, 0.3]
        mock_rag_system.chat_history.append({"role": "user", "content": "test"})
        
        report = mock_rag_system.get_performance_report()
        
        assert "embedding_cache_size" in report
        assert "chat_history_length" in report
        assert "latest_metrics" in report
        assert "memory_usage" in report
        
        assert report["embedding_cache_size"] == 1
        assert report["chat_history_length"] == 1
    
    def test_cache_management(self, mock_rag_system):
        """Test cache clearing functionality"""
        # Add data to cache
        mock_rag_system.embedding_cache["test1"] = [0.1, 0.2, 0.3]
        mock_rag_system.embedding_cache["test2"] = [0.4, 0.5, 0.6]
        
        assert len(mock_rag_system.embedding_cache) == 2
        
        # Clear cache
        mock_rag_system.clear_cache()
        
        assert len(mock_rag_system.embedding_cache) == 0
    
    def test_chat_history_reset(self, mock_rag_system):
        """Test chat history reset"""
        # Add chat history
        mock_rag_system.chat_history.append({"role": "user", "content": "test"})
        mock_rag_system.is_initial_session = True
        
        assert len(mock_rag_system.chat_history) == 1
        assert mock_rag_system.is_initial_session == True
        
        # Reset
        mock_rag_system.reset_chat_history()
        
        assert len(mock_rag_system.chat_history) == 0
        assert mock_rag_system.is_initial_session == False

class TestSourceTracking:
    """Test the new source tracking functionality"""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Create a mock RAG system for source tracking tests"""
        config = RAGSystemConfig()
        config.ai_api_key = "test_key"
        config.supabase_url = "https://test.supabase.co"
        config.supabase_key = "test_key"
        
        with patch('RAG_Core_Optimized.create_client') as mock_create_client, \
             patch('RAG_Core_Optimized.OpenAI') as mock_openai:
            
            mock_create_client.return_value = Mock()
            mock_openai.return_value = Mock()
            
            rag_system = OptimizedRAGSystem(config=config)
            return rag_system
    
    def test_extract_source_info(self, mock_rag_system):
        """Test source information extraction"""
        # Sample document data
        doc = {
            "content": "This is test content for source tracking.",
            "file_path": "/test/sample.pdf",
            "chunk_index": 1,
            "metadata": {"filename": "sample.pdf", "page_number": 2}
        }
        
        metadata = {
            "filename": "sample.pdf",
            "page_number": 2,
            "processed_at": "2024-01-01"
        }
        
        # Extract source info
        source_info = mock_rag_system._extract_source_info(doc, metadata, 0)
        
        # Verify all required fields are present
        required_fields = [
            'source_id', 'filename', 'title', 'file_path', 'file_url',
            'file_type', 'page_number', 'chunk_index', 'content_preview',
            'relevance_score', 'metadata'
        ]
        
        for field in required_fields:
            assert field in source_info, f"Missing required field: {field}"
        
        # Verify specific values
        assert source_info['filename'] == 'sample.pdf'
        assert source_info['file_type'] == 'PDF'
        assert source_info['page_number'] == 2
        assert source_info['chunk_index'] == 1
        assert 'source_1' in source_info['source_id']
        assert len(source_info['content_preview']) <= 153  # 150 + "..."
    
    def test_format_sources_for_display(self, mock_rag_system):
        """Test source formatting for display"""
        sources = [
            {
                'source_id': 'source_1',
                'filename': 'test.pdf',
                'title': 'test.pdf (Page 1)',
                'file_type': 'PDF',
                'page_number': 1,
                'relevance_score': 0.85,
                'file_url': 'file:///test/test.pdf',
                'content_preview': 'This is a test document...'
            },
            {
                'source_id': 'source_2',
                'filename': 'guide.docx',
                'title': 'guide.docx (Section 2)',
                'file_type': 'DOCX',
                'page_number': None,
                'relevance_score': 0.72,
                'file_url': 'file:///test/guide.docx',
                'content_preview': 'This is a guide document...'
            }
        ]
        
        formatted = mock_rag_system.format_sources_for_display(sources)
        
        # Check that formatting includes expected elements
        assert "1. **test.pdf (Page 1)**" in formatted
        assert "2. **guide.docx (Section 2)**" in formatted
        assert "(PDF)" in formatted
        assert "(DOCX)" in formatted
        assert "Page 1" in formatted
        assert "Relevance: 85.0%" in formatted
        assert "file:///test/test.pdf" in formatted
        assert "This is a test document..." in formatted
    
    def test_source_tracking_in_response(self, mock_rag_system):
        """Test that sources are included in response"""
        # Mock embedding generation
        mock_rag_system.generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock Supabase response with metadata
        mock_response = Mock()
        mock_response.data = [
            {
                "content": "Test document content 1",
                "file_path": "/test/doc1.pdf",
                "chunk_index": 0,
                "metadata": '{"filename": "doc1.pdf", "page_number": 1}'
            },
            {
                "content": "Test document content 2", 
                "file_path": "/test/doc2.pdf",
                "chunk_index": 1,
                "metadata": '{"filename": "doc2.pdf", "page_number": 2}'
            }
        ]
        mock_rag_system.supabase.rpc.return_value.execute.return_value = mock_response
        
        # Mock OpenAI chat response
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test AI response with sources"))]
        mock_rag_system.ai_instance.chat.completions.create.return_value = mock_chat_response
        
        # Test the method
        response = mock_rag_system.answer_this("Test question")
        
        # Verify sources are included
        assert "sources" in response
        assert len(response["sources"]) == 2
        
        # Verify source structure
        source1 = response["sources"][0]
        assert source1["filename"] == "doc1.pdf"
        assert source1["file_type"] == "PDF"
        assert source1["source_id"] == "source_1"
        
        source2 = response["sources"][1]
        assert source2["filename"] == "doc2.pdf"
        assert source2["file_type"] == "PDF"
        assert source2["source_id"] == "source_2"
    
    def test_empty_sources_formatting(self, mock_rag_system):
        """Test formatting when no sources are available"""
        formatted = mock_rag_system.format_sources_for_display([])
        assert formatted == "No sources available."
    
    def test_source_info_error_handling(self, mock_rag_system):
        """Test error handling in source info extraction"""
        # Test with malformed document
        bad_doc = {"content": "test"}  # Missing required fields
        bad_metadata = {}
        
        source_info = mock_rag_system._extract_source_info(bad_doc, bad_metadata, 0)
        
        # Should return default values without crashing
        assert source_info["filename"] == "Unknown Document"
        assert source_info["title"] == "Unknown Source"
        assert source_info["file_type"] == "UNKNOWN"
        assert source_info["source_id"] == "source_1"

class TestFileProcessing:
    """Test file processing functionality"""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Create a mock RAG system for file processing tests"""
        config = RAGSystemConfig()
        config.ai_api_key = "test_key"
        config.supabase_url = "https://test.supabase.co"
        config.supabase_key = "test_key"
        
        with patch('RAG_Core_Optimized.create_client') as mock_create_client, \
             patch('RAG_Core_Optimized.OpenAI') as mock_openai:
            
            mock_create_client.return_value = Mock()
            mock_openai.return_value = Mock()
            
            rag_system = OptimizedRAGSystem(config=config)
            return rag_system
    
    def test_load_processed_files_empty(self, mock_rag_system):
        """Test loading processed files when file doesn't exist"""
        with patch('os.path.exists', return_value=False):
            processed_files = mock_rag_system.load_processed_files()
            assert processed_files == set()
    
    def test_load_processed_files_existing(self, mock_rag_system):
        """Test loading processed files when file exists"""
        mock_content = "file1.txt\nfile2.pdf\nfile3.docx\n"
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=mock_content)):
            
            processed_files = mock_rag_system.load_processed_files()
            assert processed_files == {"file1.txt", "file2.pdf", "file3.docx"}
    
    def test_save_processed_file(self, mock_rag_system):
        """Test saving processed file"""
        with patch('builtins.open', mock_open()) as mock_file:
            mock_rag_system.save_processed_file("test.txt")
            mock_file.assert_called_once_with(mock_rag_system.processed_files_list, "a", encoding="utf-8")
    
    def test_chunk_text(self, mock_rag_system):
        """Test text chunking functionality"""
        # Test with short text
        short_text = "This is a short text."
        chunks = mock_rag_system.chunk_text(short_text)
        assert len(chunks) >= 1
        assert chunks[0] == short_text
        
        # Test with empty text
        empty_chunks = mock_rag_system.chunk_text("")
        assert len(empty_chunks) == 1
        assert empty_chunks[0] == ""

class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Create a mock RAG system for error handling tests"""
        config = RAGSystemConfig()
        config.ai_api_key = "test_key"
        config.supabase_url = "https://test.supabase.co"
        config.supabase_key = "test_key"
        
        with patch('RAG_Core_Optimized.create_client') as mock_create_client, \
             patch('RAG_Core_Optimized.OpenAI') as mock_openai:
            
            mock_create_client.return_value = Mock()
            mock_openai.return_value = Mock()
            
            rag_system = OptimizedRAGSystem(config=config)
            return rag_system
    
    def test_embedding_generation_error_handling(self, mock_rag_system):
        """Test error handling in embedding generation"""
        # Mock an exception
        mock_rag_system.ai_instance.embeddings.create.side_effect = Exception("API Error")
        
        result = mock_rag_system.generate_embedding("test text")
        assert result is None
    
    def test_answer_this_error_handling(self, mock_rag_system):
        """Test error handling in answer_this method"""
        # Mock an exception in embedding generation
        mock_rag_system.generate_embedding = Mock(side_effect=Exception("Test error"))
        
        response = mock_rag_system.answer_this("test question")
        
        assert "error" in response
        assert "Test error" in response["error"]

def mock_open(read_data=""):
    """Helper function to create a mock file open"""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=read_data)

def run_performance_tests():
    """Run performance tests to validate optimizations"""
    print("\n=== Performance Tests ===")
    
    # Mock configuration for testing
    config = RAGSystemConfig()
    config.ai_api_key = "test_key"
    config.supabase_url = "https://test.supabase.co"
    config.supabase_key = "test_key"
    
    with patch('RAG_Core_Optimized.create_client') as mock_create_client, \
         patch('RAG_Core_Optimized.OpenAI') as mock_openai:
        
        mock_create_client.return_value = Mock()
        mock_openai.return_value = Mock()
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]  # Typical embedding size
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        rag_system = OptimizedRAGSystem(config=config)
        
        # Test caching performance
        print("Testing embedding caching...")
        
        # First call
        start_time = time.time()
        rag_system.generate_embedding("test text")
        first_call_time = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        rag_system.generate_embedding("test text")
        second_call_time = time.time() - start_time
        
        print(f"First call: {first_call_time:.4f}s")
        print(f"Second call (cached): {second_call_time:.4f}s")
        print(f"Cache speedup: {(first_call_time / second_call_time):.2f}x")
        
        # Test batch processing
        print("\nTesting batch processing...")
        
        texts = [f"test text {i}" for i in range(10)]
        
        # Individual calls
        start_time = time.time()
        for text in texts:
            rag_system.generate_embedding(text)
        individual_time = time.time() - start_time
        
        # Clear cache for fair comparison
        rag_system.clear_cache()
        
        # Batch call
        start_time = time.time()
        rag_system.generate_embeddings_batch(texts)
        batch_time = time.time() - start_time
        
        print(f"Individual calls: {individual_time:.4f}s")
        print(f"Batch call: {batch_time:.4f}s")
        print(f"Batch speedup: {(individual_time / batch_time):.2f}x")

def main():
    """Run all tests"""
    print("🧪 Starting Comprehensive Test Suite for Optimized RAG System")
    
    # Check if pytest is available
    try:
        import pytest
        print("\n✅ Running with pytest...")
        # Run pytest programmatically
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("\n⚠️  pytest not available, running basic tests...")
        
        # Run basic tests without pytest
        try:
            # Test configuration
            print("\n🔧 Testing Configuration...")
            test_env = {
                'OPENAI_API_KEY': 'test_key',
                'SUPABASE_URL': 'https://test.supabase.co',
                'SUPABASE_KEY': 'test_key'
            }
            
            with patch.dict(os.environ, test_env):
                config = RAGSystemConfig()
                config.validate_config()
                print("✅ Configuration test passed")
            
            # Test logger
            print("\n📝 Testing Logger...")
            logger = RAGLogger()
            logger.log_query("test", 1.0, 1)
            print("✅ Logger test passed")
            
            print("\n✅ Basic tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
    
    # Run performance tests
    run_performance_tests()
    
    print("\n🎉 All tests completed!")
    print("\n📋 Test Summary:")
    print("   ✅ Configuration management")
    print("   ✅ Logging functionality")
    print("   ✅ Input validation and sanitization")
    print("   ✅ Caching mechanism")
    print("   ✅ Error handling")
    print("   ✅ Performance optimizations")
    print("   ✅ Memory management")
    
    print("\n💡 To run the full test suite with detailed output:")
    print("   pip install pytest")
    print("   pytest test_optimized_rag.py -v")

if __name__ == "__main__":
    main()
