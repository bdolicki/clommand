#!/usr/bin/env python3

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from clommand import ChatHistory, ClaudeREPL


class TestChatHistory:
    
    def setup_method(self):
        """Setup temporary directory for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_generate_chat_id_format(self):
        """Test chat ID generation format"""
        chat = ChatHistory()
        chat_id = chat._generate_chat_id()
        
        # Should be timestamp_hash format
        parts = chat_id.split('_')
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 8  # hash
    
    def test_generate_title_from_message(self):
        """Test title generation from user message"""
        chat = ChatHistory()
        
        # Test normal message
        title = chat._generate_title("What is the meaning of life?")
        assert title == "what-the-meaning-life"
        
        # Test with short words filtered out
        title = chat._generate_title("How do I get to the store?")
        assert title == "how-get-the-store"
        
        # Test empty message
        title = chat._generate_title("")
        assert title == "untitled-chat"
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        chat = ChatHistory()
        
        # Test special characters
        sanitized = chat._sanitize_filename("test/file:name?")
        assert sanitized == "test-file-name-"
        
        # Test long filename
        long_name = "a" * 100
        sanitized = chat._sanitize_filename(long_name)
        assert len(sanitized) == 50
    
    def test_add_message_and_auto_title(self):
        """Test adding messages and auto title generation"""
        chat = ChatHistory()
        
        # Add first user message
        chat.add_message("user", "What is Python?")
        assert len(chat.messages) == 1
        assert chat.title == ""  # Title not set yet
        
        # Add assistant response - should trigger title generation
        chat.add_message("assistant", "Python is a programming language.")
        assert len(chat.messages) == 2
        assert chat.title == "what-python"
        assert "python" in chat.filename.lower()
    
    def test_save_and_load_chat_history(self):
        """Test saving and loading chat history"""
        # Create chat with messages
        chat = ChatHistory()
        chat.add_message("user", "Hello world")
        chat.add_message("assistant", "Hi there!")
        
        # Save to file
        saved_path = chat.save_explicitly()
        assert os.path.exists(saved_path)
        
        # Load from file
        loaded_chat = ChatHistory.load_from_file(saved_path)
        assert len(loaded_chat.messages) == 2
        assert loaded_chat.messages[0]["content"] == "Hello world"
        assert loaded_chat.messages[1]["content"] == "Hi there!"
    
    def test_list_available_chats(self):
        """Test listing available chat files"""
        # Create chat history directory and files
        os.makedirs("chat_history", exist_ok=True)
        
        # Create test files
        with open("chat_history/test1.txt", "w") as f:
            f.write("test")
        with open("chat_history/test2.txt", "w") as f:
            f.write("test")
        with open("chat_history/not_txt.log", "w") as f:
            f.write("test")
        
        chats = ChatHistory.list_available_chats()
        assert len(chats) == 2
        assert "test1.txt" in chats
        assert "test2.txt" in chats
        assert "not_txt.log" not in chats


class TestClaudeREPL:
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("clommand.load_dotenv")
    def test_init_without_api_key(self, mock_load_dotenv):
        """Test initialization without API key"""
        mock_load_dotenv.return_value = None
        repl = ClaudeREPL()
        assert repl.client is None
        assert repl.running is True
        assert len(repl.commands) == 7
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("clommand.anthropic.Anthropic")
    def test_init_with_api_key(self, mock_anthropic):
        """Test initialization with API key"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        repl = ClaudeREPL()
        assert repl.client == mock_client
        mock_anthropic.assert_called_once_with(api_key="test_key")
    
    def test_get_prompt_untitled(self):
        """Test prompt generation for untitled chat"""
        repl = ClaudeREPL()
        prompt = repl._get_prompt()
        assert prompt == "untitled-chat> "
    
    def test_get_prompt_titled(self):
        """Test prompt generation for titled chat"""
        repl = ClaudeREPL()
        repl.current_chat.title = "test-chat"
        prompt = repl._get_prompt()
        assert prompt == "test-chat> "
    
    def test_get_completions_commands(self):
        """Test command completion"""
        repl = ClaudeREPL()
        
        # Test completing /sa -> /save
        completions = repl._get_completions("/sa", "/sa")
        assert "/save" in completions
        
        # Test completing /re -> /resume
        completions = repl._get_completions("/re", "/re")
        assert "/resume" in completions
    
    def test_get_completions_resume_files(self):
        """Test file completion for /resume command"""
        repl = ClaudeREPL()
        
        # Create test chat files
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/test-chat.txt", "w") as f:
            f.write("test")
        with open("chat_history/another-chat.txt", "w") as f:
            f.write("test")
        
        # Test completion after /resume 
        completions = repl._get_completions("test", "/resume test")
        assert "test-chat.txt" in completions
        assert "another-chat.txt" not in completions
    
    def test_handle_command_quit(self):
        """Test /quit command"""
        repl = ClaudeREPL()
        result = repl._handle_command("/quit")
        assert result is True
        assert repl.running is False
    
    def test_handle_command_new(self):
        """Test /new command"""
        repl = ClaudeREPL()
        old_chat_id = repl.current_chat.chat_id
        
        result = repl._handle_command("/new")
        assert result is True
        assert repl.current_chat.chat_id != old_chat_id
    
    def test_handle_command_save(self, capsys):
        """Test /save command"""
        repl = ClaudeREPL()
        repl.current_chat.add_message("user", "test")
        
        result = repl._handle_command("/save")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Chat saved to:" in captured.out
    
    def test_handle_command_list_empty(self, capsys):
        """Test /list command with no chats"""
        repl = ClaudeREPL()
        
        result = repl._handle_command("/list")
        assert result is True
        
        captured = capsys.readouterr()
        assert "No saved chat histories found" in captured.out
    
    def test_handle_command_list_with_chats(self, capsys):
        """Test /list command with existing chats"""
        repl = ClaudeREPL()
        
        # Create test chat files
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/test-chat.txt", "w") as f:
            f.write("test")
        
        result = repl._handle_command("/list")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Available chat histories:" in captured.out
        assert "test-chat.txt" in captured.out
    
    def test_handle_command_resume_success(self, capsys):
        """Test /resume command success"""
        repl = ClaudeREPL()
        
        # Create a test chat file
        os.makedirs("chat_history", exist_ok=True)
        test_content = """# Chat: test-chat
# ID: test_id
# Created: 2023-01-01T00:00:00

[2023-01-01T00:00:00] USER: Hello

[2023-01-01T00:00:01] ASSISTANT: Hi there!

"""
        with open("chat_history/test-chat.txt", "w") as f:
            f.write(test_content)
        
        result = repl._handle_command("/resume test-chat.txt")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Resumed chat: test-chat" in captured.out
        assert "Messages loaded: 2" in captured.out
        assert "Last conversation turn" in captured.out
    
    def test_handle_command_resume_file_not_found(self, capsys):
        """Test /resume command with non-existent file"""
        repl = ClaudeREPL()
        
        result = repl._handle_command("/resume nonexistent.txt")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Chat file not found" in captured.out
    
    def test_handle_command_help(self, capsys):
        """Test /help command"""
        repl = ClaudeREPL()
        
        result = repl._handle_command("/help")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Available commands:" in captured.out
        assert "/save" in captured.out
        assert "/quit" in captured.out
    
    def test_handle_command_system(self, mocker):
        """Test /system command"""
        repl = ClaudeREPL()
        
        # Mock the editor opening
        mock_open_editor = mocker.patch.object(repl, '_open_system_prompt_in_editor')
        
        result = repl._handle_command("/system")
        assert result is True
        mock_open_editor.assert_called_once()
    
    def test_handle_command_unknown(self, capsys):
        """Test unknown command"""
        repl = ClaudeREPL()
        
        result = repl._handle_command("/unknown")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("clommand.anthropic.Anthropic")
    def test_get_claude_response_success(self, mock_anthropic):
        """Test successful Claude API response"""
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        repl = ClaudeREPL()
        response = repl._get_claude_response("Test input")
        
        assert response == "Test response"
        mock_client.messages.create.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("clommand.load_dotenv")
    def test_get_claude_response_no_client(self, mock_load_dotenv):
        """Test Claude response without client"""
        mock_load_dotenv.return_value = None
        repl = ClaudeREPL()
        response = repl._get_claude_response("Test input")
        assert "Error: Claude client not initialized" in response
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("clommand.anthropic.Anthropic")
    def test_get_claude_response_api_error(self, mock_anthropic):
        """Test Claude API error handling"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        repl = ClaudeREPL()
        response = repl._get_claude_response("Test input")
        
        assert "Error getting Claude response" in response
        assert "API Error" in response


    def test_system_prompt_functionality(self):
        """Test system prompt file management"""
        repl = ClaudeREPL()
        
        # Test that system prompt file is created
        assert os.path.exists(repl.system_prompt_file)
        
        # Test reading system prompt
        prompt = repl._get_system_prompt()
        assert "Claude" in prompt
        assert "Anthropic" in prompt
        
        # Test writing custom system prompt
        custom_prompt = "You are a helpful coding assistant."
        with open(repl.system_prompt_file, 'w') as f:
            f.write(custom_prompt)
        
        # Test reading custom prompt
        read_prompt = repl._get_system_prompt()
        assert read_prompt == custom_prompt
    
    def test_is_command_available(self):
        """Test command availability check"""
        repl = ClaudeREPL()
        
        # Test with a command that should exist
        assert repl._is_command_available('python') or repl._is_command_available('python3')
        
        # Test with a command that shouldn't exist
        assert not repl._is_command_available('nonexistent_command_12345')
    
    def test_system_prompt_integration_with_api(self, mocker):
        """Test that system prompt is included in API calls"""
        repl = ClaudeREPL()
        
        # Create custom system prompt
        custom_prompt = "You are a test assistant."
        with open(repl.system_prompt_file, 'w') as f:
            f.write(custom_prompt)
        
        # Mock the API client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        repl.client = mock_client
        
        # Make a request
        response = repl._get_claude_response("Test input")
        
        # Verify system prompt was included
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['system'] == custom_prompt


class TestIntegration:
    
    def setup_method(self):
        """Setup temporary directory for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after integration tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_full_chat_workflow(self):
        """Test complete chat workflow: create -> save -> resume -> append"""
        # Create chat and add messages
        chat = ChatHistory()
        chat.add_message("user", "What is machine learning?")
        chat.add_message("assistant", "Machine learning is a subset of AI.")
        
        # Save chat
        saved_path = chat.save_explicitly()
        assert os.path.exists(saved_path)
        original_filename = chat.filename
        
        # Load chat
        loaded_chat = ChatHistory.load_from_file(saved_path)
        assert len(loaded_chat.messages) == 2
        assert loaded_chat.title == "what-machine-learning"
        assert loaded_chat.messages[0]["content"] == "What is machine learning?"
        assert loaded_chat.messages[1]["content"] == "Machine learning is a subset of AI."
        
        # Verify filename is preserved
        assert loaded_chat.filename == original_filename
        
        # Add new message to resumed chat
        loaded_chat.add_message("user", "Can you explain neural networks?")
        loaded_chat.add_message("assistant", "Neural networks are computing systems inspired by biological neural networks.")
        
        # Verify new messages are saved to the same file
        assert len(loaded_chat.messages) == 4
        assert loaded_chat.filename == original_filename
        
        # Verify file contains all messages
        reloaded_chat = ChatHistory.load_from_file(saved_path)
        assert len(reloaded_chat.messages) == 4
        assert reloaded_chat.messages[2]["content"] == "Can you explain neural networks?"
    
    def test_chat_file_naming_and_renaming(self):
        """Test chat file naming and automatic renaming"""
        chat = ChatHistory()
        
        # Initial filename should be timestamp-based
        initial_filename = chat.filename
        assert initial_filename.endswith('.txt')
        
        # Add messages to trigger title generation
        chat.add_message("user", "Tell me about Python programming")
        chat.add_message("assistant", "Python is a versatile language.")
        
        # Filename should now be content-based
        assert chat.filename != initial_filename
        assert "python-programming" in chat.filename.lower()
        
        # File should exist in chat_history
        assert os.path.exists(os.path.join("chat_history", chat.filename))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])