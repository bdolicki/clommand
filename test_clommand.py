#!/usr/bin/env python3

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from clommand import ChatHistory, ClaudeChatbot


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


class TestClaudeChatbot:
    
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
        chatbot = ClaudeChatbot()
        assert chatbot.client is None
        assert chatbot.running is True
        assert len(chatbot.commands) == 9
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("clommand.anthropic.Anthropic")
    def test_init_with_api_key(self, mock_anthropic):
        """Test initialization with API key"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        assert chatbot.client == mock_client
        mock_anthropic.assert_called_once_with(api_key="test_key")
    
    def test_get_prompt_untitled(self):
        """Test prompt generation for untitled chat"""
        chatbot = ClaudeChatbot()
        prompt = chatbot._get_prompt()
        assert prompt == "untitled-chat> "
    
    def test_get_prompt_titled(self):
        """Test prompt generation for titled chat"""
        chatbot = ClaudeChatbot()
        chatbot.current_chat.title = "test-chat"
        prompt = chatbot._get_prompt()
        assert prompt == "test-chat> "
    
    def test_get_completions_commands(self):
        """Test command completion"""
        chatbot = ClaudeChatbot()
        
        # Test completing /sa -> /save
        completions = chatbot._get_completions("/sa", "/sa")
        assert "/save" in completions
        
        # Test completing /re -> /resume
        completions = chatbot._get_completions("/re", "/re")
        assert "/resume" in completions
    
    def test_get_completions_resume_files(self):
        """Test file completion for /resume command"""
        chatbot = ClaudeChatbot()
        
        # Create test chat files
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/test-chat.txt", "w") as f:
            f.write("test")
        with open("chat_history/another-chat.txt", "w") as f:
            f.write("test")
        
        # Test completion after /resume 
        completions = chatbot._get_completions("test", "/resume test")
        assert "test-chat.txt" in completions
        assert "another-chat.txt" not in completions
    
    def test_get_completions_model_names(self):
        """Test model completion for /model command"""
        chatbot = ClaudeChatbot()
        
        # Test completion after /model 
        completions = chatbot._get_completions("ha", "/model ha")
        assert "haiku" in completions
        assert "sonnet" not in completions
        assert "opus" not in completions
        
        # Test completion for all models
        completions = chatbot._get_completions("", "/model ")
        assert "haiku" in completions
        assert "sonnet" in completions
        assert "opus" in completions
    
    def test_get_completions_log_levels(self):
        """Test log level completion for /log command"""
        chatbot = ClaudeChatbot()
        
        # Test completion after /log 
        completions = chatbot._get_completions("de", "/log de")
        assert "debug" in completions
        assert "info" not in completions
        
        # Test completion for all log levels
        completions = chatbot._get_completions("", "/log ")
        assert "trace" in completions
        assert "debug" in completions
        assert "info" in completions
        assert "warning" in completions
        assert "error" in completions
    
    def test_handle_command_quit(self):
        """Test /quit command"""
        chatbot = ClaudeChatbot()
        result = chatbot._handle_command("/quit")
        assert result is True
        assert chatbot.running is False
    
    def test_handle_command_new(self):
        """Test /new command"""
        chatbot = ClaudeChatbot()
        old_chat_id = chatbot.current_chat.chat_id
        
        result = chatbot._handle_command("/new")
        assert result is True
        assert chatbot.current_chat.chat_id != old_chat_id
    
    def test_handle_command_save(self, capsys):
        """Test /save command"""
        chatbot = ClaudeChatbot()
        chatbot.current_chat.add_message("user", "test")
        
        result = chatbot._handle_command("/save")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Chat saved to:" in captured.out
    
    def test_handle_command_list_empty(self, capsys):
        """Test /list command with no chats"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/list")
        assert result is True
        
        captured = capsys.readouterr()
        assert "No saved chat histories found" in captured.out
    
    def test_handle_command_list_with_chats(self, capsys):
        """Test /list command with existing chats"""
        chatbot = ClaudeChatbot()
        
        # Create test chat files
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/test-chat.txt", "w") as f:
            f.write("test")
        
        result = chatbot._handle_command("/list")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Available chat histories:" in captured.out
        assert "test-chat.txt" in captured.out
    
    def test_handle_command_resume_success(self, capsys):
        """Test /resume command success"""
        chatbot = ClaudeChatbot()
        
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
        
        result = chatbot._handle_command("/resume test-chat.txt")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Resumed chat: test-chat" in captured.out
        assert "Messages loaded: 2" in captured.out
        assert "Last conversation turn" in captured.out
    
    def test_handle_command_resume_file_not_found(self, capsys):
        """Test /resume command with non-existent file"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/resume nonexistent.txt")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Chat file not found" in captured.out
    
    def test_handle_command_help(self, capsys):
        """Test /help command"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/help")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Available commands:" in captured.out
        assert "/save" in captured.out
        assert "/quit" in captured.out
    
    def test_handle_command_system(self, mocker):
        """Test /system command"""
        chatbot = ClaudeChatbot()
        
        # Mock the editor opening
        mock_open_editor = mocker.patch.object(chatbot, '_open_system_prompt_in_editor')
        
        result = chatbot._handle_command("/system")
        assert result is True
        mock_open_editor.assert_called_once()
    
    def test_handle_command_model_show(self, capsys):
        """Test /model command to show current model"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/model")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Current model: haiku" in captured.out
        assert "claude-3-5-haiku-20241022" in captured.out
    
    def test_handle_command_model_switch(self, capsys):
        """Test /model command to switch models"""
        chatbot = ClaudeChatbot()
        
        # Switch to sonnet
        result = chatbot._handle_command("/model sonnet")
        assert result is True
        assert chatbot.current_model == "sonnet"
        
        captured = capsys.readouterr()
        assert "Switched to model: sonnet" in captured.out
        assert "claude-sonnet-4-20250514" in captured.out
        
        # Switch to opus
        result = chatbot._handle_command("/model opus")
        assert result is True
        assert chatbot.current_model == "opus"
    
    def test_handle_command_model_invalid(self, capsys):
        """Test /model command with invalid model"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/model invalid")
        assert result is True
        assert chatbot.current_model == "haiku"  # Should remain unchanged
        
        captured = capsys.readouterr()
        assert "Unknown model: invalid" in captured.out
        assert "Available models:" in captured.out
    
    def test_handle_command_log_show(self, capsys):
        """Test /log command to show current log level"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/log")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Current log level: info" in captured.out
        assert "Available levels:" in captured.out
    
    def test_handle_command_log_switch(self, capsys):
        """Test /log command to switch log levels"""
        chatbot = ClaudeChatbot()
        
        # Switch to trace
        result = chatbot._handle_command("/log trace")
        assert result is True
        assert chatbot.current_log_level == "trace"
        
        captured = capsys.readouterr()
        assert "Log level set to: trace" in captured.out
        
        # Switch to debug
        result = chatbot._handle_command("/log debug")
        assert result is True
        assert chatbot.current_log_level == "debug"
        
        # Switch to error
        result = chatbot._handle_command("/log error")
        assert result is True
        assert chatbot.current_log_level == "error"
    
    def test_handle_command_log_invalid(self, capsys):
        """Test /log command with invalid level"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/log invalid")
        assert result is True
        assert chatbot.current_log_level == "info"  # Should remain unchanged
        
        captured = capsys.readouterr()
        assert "Unknown log level: invalid" in captured.out
        assert "Available levels:" in captured.out
    
    def test_logging_setup(self):
        """Test that logging is properly configured"""
        chatbot = ClaudeChatbot()
        
        # Check that logger exists
        assert hasattr(chatbot, 'logger')
        assert chatbot.logger.name == 'clommand'
        
        # Check that logs directory exists
        assert os.path.exists("logs")
        
        # Check that log level is set correctly
        assert chatbot.current_log_level == "info"
    
    def test_trace_logging_functionality(self):
        """Test that trace level logging includes full API payloads"""
        chatbot = ClaudeChatbot()
        
        # Switch to trace level
        chatbot.current_log_level = "trace"
        chatbot._setup_logging()
        
        # Check that trace method is available
        assert hasattr(chatbot.logger, 'trace')
        
        # Check that trace level is enabled
        assert chatbot.logger.isEnabledFor(5)  # TRACE level
    
    def test_handle_command_unknown(self, capsys):
        """Test unknown command"""
        chatbot = ClaudeChatbot()
        
        result = chatbot._handle_command("/unknown")
        assert result is True
        
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("clommand.anthropic.Anthropic")
    def test_get_ai_response_success(self, mock_anthropic):
        """Test successful AI API response"""
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        response = chatbot._get_ai_response("Test input")
        
        assert response == "Test response"
        mock_client.messages.create.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("clommand.load_dotenv")
    def test_get_ai_response_no_client(self, mock_load_dotenv):
        """Test AI response without client"""
        mock_load_dotenv.return_value = None
        chatbot = ClaudeChatbot()
        response = chatbot._get_ai_response("Test input")
        assert "client not initialized" in response
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("clommand.anthropic.Anthropic")
    def test_get_ai_response_api_error(self, mock_anthropic):
        """Test AI API error handling"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        response = chatbot._get_ai_response("Test input")
        
        assert "Error getting AI response" in response
        assert "API Error" in response


    def test_system_prompt_functionality(self):
        """Test system prompt file management"""
        chatbot = ClaudeChatbot()
        
        # Test that system prompt file is created
        assert os.path.exists(chatbot.system_prompt_file)
        
        # Test reading system prompt
        prompt = chatbot._get_system_prompt()
        assert "Claude" in prompt
        assert "Anthropic" in prompt
        
        # Test writing custom system prompt
        custom_prompt = "You are a helpful coding assistant."
        with open(chatbot.system_prompt_file, 'w') as f:
            f.write(custom_prompt)
        
        # Test reading custom prompt
        read_prompt = chatbot._get_system_prompt()
        assert read_prompt == custom_prompt
    
    def test_is_command_available(self):
        """Test command availability check"""
        chatbot = ClaudeChatbot()
        
        # Test with a command that should exist
        assert chatbot._is_command_available('python') or chatbot._is_command_available('python3')
        
        # Test with a command that shouldn't exist
        assert not chatbot._is_command_available('nonexistent_command_12345')
    
    def test_system_prompt_integration_with_api(self, mocker):
        """Test that system prompt is included in API calls"""
        chatbot = ClaudeChatbot()
        
        # Create custom system prompt
        custom_prompt = "You are a test assistant."
        with open(chatbot.system_prompt_file, 'w') as f:
            f.write(custom_prompt)
        
        # Mock the Anthropic client (haiku model uses Anthropic by default)
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        chatbot.anthropic_client = mock_client
        
        # Make a request
        response = chatbot._get_ai_response("Test input")
        
        # Verify system prompt was included
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['system'] == custom_prompt
    
    def test_model_selection_integration_with_api(self, mocker):
        """Test that selected model is used in API calls"""
        chatbot = ClaudeChatbot()
        
        # Switch to opus model
        chatbot.current_model = "opus"
        
        # Mock the Anthropic client (haiku model uses Anthropic by default)
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        chatbot.anthropic_client = mock_client
        
        # Make a request
        response = chatbot._get_ai_response("Test input")
        
        # Verify correct model was used
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == "claude-opus-4-20250514"


class TestOpenAIIntegration:
    
    def setup_method(self):
        """Setup for OpenAI integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after OpenAI integration tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"})
    @patch("clommand.openai.OpenAI")
    def test_openai_client_initialization(self, mock_openai):
        """Test OpenAI client is properly initialized"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        assert chatbot.openai_client == mock_client
        mock_openai.assert_called_once_with(api_key="test_openai_key")
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key", "OPENAI_API_KEY": "test_openai_key"})
    @patch("clommand.anthropic.Anthropic")
    @patch("clommand.openai.OpenAI")
    def test_dual_provider_initialization(self, mock_openai, mock_anthropic):
        """Test both providers are initialized when both API keys are present"""
        mock_anthropic_client = Mock()
        mock_openai_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_openai.return_value = mock_openai_client
        
        chatbot = ClaudeChatbot()
        assert chatbot.anthropic_client == mock_anthropic_client
        assert chatbot.openai_client == mock_openai_client
        assert chatbot.client == mock_anthropic_client  # Primary client should be Anthropic
    
    def test_model_provider_detection(self):
        """Test correct provider detection for different models"""
        chatbot = ClaudeChatbot()
        
        # Test Anthropic models
        assert chatbot._get_model_provider("haiku") == "anthropic"
        assert chatbot._get_model_provider("sonnet") == "anthropic"
        assert chatbot._get_model_provider("opus") == "anthropic"
        
        # Test OpenAI models
        assert chatbot._get_model_provider("gpt4") == "openai"
        assert chatbot._get_model_provider("gpt4-mini") == "openai"
        assert chatbot._get_model_provider("o1") == "openai"
        assert chatbot._get_model_provider("o1-mini") == "openai"
        assert chatbot._get_model_provider("o1-pro") == "openai"
        assert chatbot._get_model_provider("o4-mini") == "openai"
    
    def test_expanded_model_options(self):
        """Test that all new OpenAI models are properly configured"""
        chatbot = ClaudeChatbot()
        
        # Verify all expected models are present
        expected_models = {
            # Anthropic
            'haiku', 'sonnet', 'opus',
            # OpenAI
            'gpt4', 'gpt4-latest', 'gpt4-mini', 
            'o1', 'o1-mini', 'o1-pro', 'o4-mini'
        }
        
        actual_models = set(chatbot.model_options.keys())
        assert actual_models == expected_models
        
        # Verify specific model mappings
        assert chatbot.model_options['gpt4'] == 'gpt-4o'
        assert chatbot.model_options['gpt4-latest'] == 'chatgpt-4o-latest'
        assert chatbot.model_options['o1'] == 'o1'
        assert chatbot.model_options['o1-pro'] == 'o1-pro'
        assert chatbot.model_options['o4-mini'] == 'o4-mini'
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("clommand.openai.OpenAI")
    def test_openai_api_call_regular_model(self, mock_openai):
        """Test OpenAI API call for regular models (non-o1)"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        chatbot.current_model = "gpt4"
        chatbot.current_chat.add_message("user", "Test input")
        
        response = chatbot._get_ai_response("Test input")
        assert response == "Test response"
        
        # Verify API call format
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args['model'] == 'gpt-4o'
        assert call_args['max_tokens'] == 1024
        assert len(call_args['messages']) == 2  # system + user message
        assert call_args['messages'][0]['role'] == 'system'
        assert call_args['messages'][1]['role'] == 'user'
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("clommand.openai.OpenAI") 
    def test_openai_api_call_o1_model(self, mock_openai):
        """Test OpenAI API call for o1 models with system prompt emulation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.completion_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        chatbot.current_model = "o1-mini"
        chatbot.current_chat.add_message("user", "Test input")
        
        response = chatbot._get_ai_response("Test input")
        assert response == "Test response"
        
        # Verify API call format for o1 model
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args['model'] == 'o1-mini'
        assert call_args['max_completion_tokens'] == 4096  # Different token param
        assert 'max_tokens' not in call_args  # Should not have max_tokens
        assert len(call_args['messages']) == 1  # No separate system message
        # System prompt should be embedded in user message
        user_message = call_args['messages'][0]['content']
        system_prompt = chatbot._get_system_prompt()
        assert user_message.startswith(system_prompt)


class TestSystemPromptEmulation:
    
    def setup_method(self):
        """Setup for system prompt emulation tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after system prompt emulation tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_emulate_system_prompt_for_o1_basic(self):
        """Test basic system prompt emulation for o1 models"""
        chatbot = ClaudeChatbot()
        system_prompt = "You are a helpful assistant."
        messages = [{"role": "user", "content": "Hello"}]
        
        result = chatbot._emulate_system_prompt_for_o1(system_prompt, messages)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == f"{system_prompt}\n\nHello"
    
    def test_emulate_system_prompt_for_o1_already_embedded(self):
        """Test system prompt emulation when prompt is already embedded"""
        chatbot = ClaudeChatbot()
        system_prompt = "You are a helpful assistant."
        embedded_content = f"{system_prompt}\n\nHello"
        messages = [{"role": "user", "content": embedded_content}]
        
        result = chatbot._emulate_system_prompt_for_o1(system_prompt, messages)
        
        assert len(result) == 1
        assert result[0]["content"] == embedded_content  # Should remain unchanged
    
    def test_emulate_system_prompt_for_o1_multiple_messages(self):
        """Test system prompt emulation with multiple messages"""
        chatbot = ClaudeChatbot()
        system_prompt = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = chatbot._emulate_system_prompt_for_o1(system_prompt, messages)
        
        assert len(result) == 3
        # Only first user message should be modified
        assert result[0]["content"] == f"{system_prompt}\n\nHello"
        assert result[1]["content"] == "Hi there!"
        assert result[2]["content"] == "How are you?"
    
    def test_clean_embedded_system_prompt_basic(self):
        """Test cleaning embedded system prompt from messages"""
        chatbot = ClaudeChatbot()
        system_prompt = "You are a helpful assistant."
        embedded_content = f"{system_prompt}\n\nHello"
        messages = [{"role": "user", "content": embedded_content}]
        
        result = chatbot._clean_embedded_system_prompt(system_prompt, messages)
        
        assert len(result) == 1
        assert result[0]["content"] == "Hello"
    
    def test_clean_embedded_system_prompt_not_embedded(self):
        """Test cleaning when system prompt is not embedded"""
        chatbot = ClaudeChatbot()
        system_prompt = "You are a helpful assistant."
        messages = [{"role": "user", "content": "Hello"}]
        
        result = chatbot._clean_embedded_system_prompt(system_prompt, messages)
        
        assert len(result) == 1
        assert result[0]["content"] == "Hello"  # Should remain unchanged
    
    def test_clean_embedded_system_prompt_multiple_messages(self):
        """Test cleaning embedded system prompt with multiple messages"""
        chatbot = ClaudeChatbot()
        system_prompt = "You are a helpful assistant."
        embedded_content = f"{system_prompt}\n\nHello"
        messages = [
            {"role": "user", "content": embedded_content},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = chatbot._clean_embedded_system_prompt(system_prompt, messages)
        
        assert len(result) == 3
        # Only first user message should be cleaned
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there!"
        assert result[2]["content"] == "How are you?"


class TestTokenExhaustionDetection:
    
    def setup_method(self):
        """Setup for token exhaustion tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after token exhaustion tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("clommand.openai.OpenAI")
    def test_token_exhaustion_detection(self, mock_openai):
        """Test detection of token exhaustion in o1 models"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""  # Empty response
        mock_response.usage = Mock()
        mock_response.usage.completion_tokens = 4000  # Near limit
        mock_response.usage.completion_tokens_details = Mock()
        mock_response.usage.completion_tokens_details.reasoning_tokens = 3900
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        chatbot.current_model = "o1-mini"
        chatbot.current_chat.add_message("user", "Complex question")
        
        response = chatbot._get_ai_response("Complex question")
        
        assert response.startswith("Error: o1-mini exhausted its token limit")
        assert "4000/4096 tokens" in response
        assert "3900 for reasoning" in response
        assert "simpler question" in response
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("clommand.openai.OpenAI")
    def test_no_token_exhaustion_normal_usage(self, mock_openai):
        """Test normal operation doesn't trigger exhaustion detection"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Normal response"
        mock_response.usage = Mock()
        mock_response.usage.completion_tokens = 500  # Well under limit
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        chatbot.current_model = "o1-mini"
        chatbot.current_chat.add_message("user", "Simple question")
        
        response = chatbot._get_ai_response("Simple question")
        
        assert response == "Normal response"
        assert not response.startswith("Error:")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("clommand.openai.OpenAI")
    def test_token_exhaustion_only_for_o1_models(self, mock_openai):
        """Test that token exhaustion detection only applies to o1 models"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""  # Empty response
        mock_response.usage = Mock()
        mock_response.usage.completion_tokens = 1000
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        chatbot.current_model = "gpt4"  # Regular model, not o1
        chatbot.current_chat.add_message("user", "Question")
        
        response = chatbot._get_ai_response("Question")
        
        # Should return empty string, not trigger exhaustion detection
        assert response == ""
        assert not response.startswith("Error: gpt4 exhausted")


class TestProviderValidation:
    
    def setup_method(self):
        """Setup for provider validation tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after provider validation tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True)
    @patch("clommand.anthropic.Anthropic")
    @patch("clommand.openai.OpenAI", side_effect=Exception("No API key"))
    def test_model_switch_validation_anthropic_only(self, mock_openai, mock_anthropic, capsys):
        """Test model switching validation when only Anthropic key is available"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        # Verify only Anthropic client is available
        assert chatbot.anthropic_client is not None
        assert chatbot.openai_client is None
        
        # Should succeed for Anthropic models
        result = chatbot._handle_command("/model sonnet")
        assert result is True
        assert chatbot.current_model == "sonnet"
        
        # Should fail for OpenAI models
        result = chatbot._handle_command("/model gpt4")
        assert result is True
        assert chatbot.current_model == "sonnet"  # Should remain unchanged
        
        captured = capsys.readouterr()
        assert "Error: Openai API key not configured" in captured.out
        assert "OPENAI_API_KEY" in captured.out
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("clommand.openai.OpenAI")
    @patch("clommand.anthropic.Anthropic", side_effect=Exception("No API key"))
    def test_model_switch_validation_openai_only(self, mock_anthropic, mock_openai, capsys):
        """Test model switching validation when only OpenAI key is available"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chatbot = ClaudeChatbot()
        # Verify only OpenAI client is available
        assert chatbot.anthropic_client is None
        assert chatbot.openai_client is not None
        
        chatbot.current_model = "gpt4"  # Start with OpenAI model
        
        # Should succeed for OpenAI models
        result = chatbot._handle_command("/model o1-mini")
        assert result is True
        assert chatbot.current_model == "o1-mini"
        
        # Should fail for Anthropic models
        result = chatbot._handle_command("/model haiku")
        assert result is True
        assert chatbot.current_model == "o1-mini"  # Should remain unchanged
        
        captured = capsys.readouterr()
        assert "Error: Anthropic API key not configured" in captured.out
        assert "ANTHROPIC_API_KEY" in captured.out


class TestModelSwitching:
    
    def setup_method(self):
        """Setup for model switching tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after model switching tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key", "OPENAI_API_KEY": "test_openai_key"})
    @patch("clommand.anthropic.Anthropic")
    @patch("clommand.openai.OpenAI")
    def test_model_switching_anthropic_to_openai_regular(self, mock_openai, mock_anthropic):
        """Test switching from Anthropic to OpenAI regular model"""
        # Setup mocks
        mock_anthropic_client = Mock()
        mock_openai_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_openai.return_value = mock_openai_client
        
        chatbot = ClaudeChatbot()
        
        # Start with Anthropic model and add some chat history
        chatbot.current_model = "haiku"
        chatbot.current_chat.add_message("user", "Hello")
        chatbot.current_chat.add_message("assistant", "Hi there!")
        
        # Switch to OpenAI regular model
        result = chatbot._handle_command("/model gpt4")
        assert result is True
        assert chatbot.current_model == "gpt4"
        
        # Test that system prompt handling works correctly for regular OpenAI model
        messages = [{"role": msg["role"], "content": msg["content"]} 
                   for msg in chatbot.current_chat.messages]
        system_prompt = "Test system prompt"
        
        # Should clean any embedded prompts and use separate system message
        cleaned_messages = chatbot._clean_embedded_system_prompt(system_prompt, messages)
        assert len(cleaned_messages) == 2
        assert cleaned_messages[0]["content"] == "Hello"  # Should be clean
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key", "OPENAI_API_KEY": "test_openai_key"})
    @patch("clommand.anthropic.Anthropic")
    @patch("clommand.openai.OpenAI")
    def test_model_switching_anthropic_to_o1(self, mock_openai, mock_anthropic):
        """Test switching from Anthropic to OpenAI o1 model"""
        # Setup mocks
        mock_anthropic_client = Mock()
        mock_openai_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_openai.return_value = mock_openai_client
        
        chatbot = ClaudeChatbot()
        
        # Start with Anthropic model and add chat history
        chatbot.current_model = "sonnet"
        chatbot.current_chat.add_message("user", "What is AI?")
        chatbot.current_chat.add_message("assistant", "AI is artificial intelligence.")
        
        # Switch to o1 model
        result = chatbot._handle_command("/model o1-mini")
        assert result is True
        assert chatbot.current_model == "o1-mini"
        
        # Test that system prompt emulation works for o1 model
        messages = [{"role": msg["role"], "content": msg["content"]} 
                   for msg in chatbot.current_chat.messages]
        system_prompt = "Test system prompt"
        
        # Should embed system prompt in first user message
        emulated_messages = chatbot._emulate_system_prompt_for_o1(system_prompt, messages)
        assert len(emulated_messages) == 2
        assert emulated_messages[0]["content"].startswith("Test system prompt")
        assert "What is AI?" in emulated_messages[0]["content"]
        assert emulated_messages[1]["content"] == "AI is artificial intelligence."
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key", "OPENAI_API_KEY": "test_openai_key"})
    @patch("clommand.anthropic.Anthropic")
    @patch("clommand.openai.OpenAI")
    def test_model_switching_o1_to_regular_openai(self, mock_openai, mock_anthropic):
        """Test switching from o1 model to regular OpenAI model"""
        # Setup mocks
        mock_anthropic_client = Mock()
        mock_openai_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_openai.return_value = mock_openai_client
        
        chatbot = ClaudeChatbot()
        system_prompt = chatbot._get_system_prompt()
        
        # Start with o1 model and simulate embedded system prompt
        chatbot.current_model = "o1"
        embedded_content = f"{system_prompt}\n\nWhat is machine learning?"
        chatbot.current_chat.add_message("user", embedded_content)
        chatbot.current_chat.add_message("assistant", "ML is a subset of AI.")
        
        # Switch to regular OpenAI model
        result = chatbot._handle_command("/model gpt4")
        assert result is True
        assert chatbot.current_model == "gpt4"
        
        # Test that embedded system prompt is cleaned for regular model
        messages = [{"role": msg["role"], "content": msg["content"]} 
                   for msg in chatbot.current_chat.messages]
        
        cleaned_messages = chatbot._clean_embedded_system_prompt(system_prompt, messages)
        assert len(cleaned_messages) == 2
        assert cleaned_messages[0]["content"] == "What is machine learning?"  # Should be clean
        assert cleaned_messages[1]["content"] == "ML is a subset of AI."
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key", "OPENAI_API_KEY": "test_openai_key"})
    @patch("clommand.anthropic.Anthropic")
    @patch("clommand.openai.OpenAI")
    def test_model_switching_o1_to_anthropic(self, mock_openai, mock_anthropic):
        """Test switching from o1 model to Anthropic model"""
        # Setup mocks
        mock_anthropic_client = Mock()
        mock_openai_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_openai.return_value = mock_openai_client
        
        chatbot = ClaudeChatbot()
        system_prompt = chatbot._get_system_prompt()
        
        # Start with o1 model with embedded system prompt
        chatbot.current_model = "o1-pro"
        embedded_content = f"{system_prompt}\n\nExplain quantum computing"
        chatbot.current_chat.add_message("user", embedded_content)
        chatbot.current_chat.add_message("assistant", "Quantum computing uses quantum mechanics.")
        
        # Switch to Anthropic model
        result = chatbot._handle_command("/model opus")
        assert result is True
        assert chatbot.current_model == "opus"
        
        # Test that Anthropic API call works correctly (it doesn't use the cleaning logic)
        # Anthropic uses separate system parameter, so embedded prompts don't interfere
        assert chatbot._get_model_provider("opus") == "anthropic"


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
        """Test chat file naming and automatic renaming with cleanup"""
        chat = ChatHistory()
        
        # Initial filename should be timestamp-based
        initial_filename = chat.filename
        initial_filepath = os.path.join("chat_history", initial_filename)
        assert initial_filename.endswith('.txt')
        
        # Add first message - this should create the initial file
        chat.add_message("user", "Tell me about Python programming")
        assert os.path.exists(initial_filepath)
        
        # Add assistant response - this should trigger title generation and cleanup
        chat.add_message("assistant", "Python is a versatile language.")
        
        # Filename should now be content-based
        final_filename = chat.filename
        final_filepath = os.path.join("chat_history", final_filename)
        assert final_filename != initial_filename
        assert "python-programming" in final_filename.lower()
        
        # New file should exist and old file should be cleaned up
        assert os.path.exists(final_filepath)
        assert not os.path.exists(initial_filepath), f"Old file {initial_filename} should be cleaned up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])