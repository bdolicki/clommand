#!/usr/bin/env python3

import os
import sys
import json
import re
import hashlib
import readline
import subprocess
import tempfile
import logging
import textwrap
import warnings
from datetime import datetime
from typing import List, Dict, Optional
import anthropic
import openai
from dotenv import load_dotenv

class ChatHistory:
    def __init__(self, chat_id: str = None):
        self.chat_id = chat_id or self._generate_chat_id()
        self.messages: List[Dict] = []
        self.filename = f"{self.chat_id}.txt"
        self.title = ""
        
    def _generate_chat_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._auto_update_title()
        self._save_continuously()
    
    def _auto_update_title(self):
        if not self.title and len(self.messages) >= 2:
            first_user_msg = next((msg["content"] for msg in self.messages if msg["role"] == "user"), "")
            if first_user_msg:
                self.title = self._generate_title(first_user_msg)
                old_filename = self.filename
                old_filepath = os.path.join("chat_history", old_filename)
                self.filename = f"{self._sanitize_filename(self.title)}.txt"
                new_filepath = os.path.join("chat_history", self.filename)
                
                # If files are different and old file exists, clean up the old file
                if os.path.exists(old_filepath) and old_filename != self.filename:
                    # Remove the old file since we'll save to the new location with meaningful name
                    os.remove(old_filepath)
    
    def _generate_title(self, first_message: str) -> str:
        words = re.findall(r'\w+', first_message.lower())
        title_words = [w for w in words if len(w) > 2][:5]
        return "-".join(title_words) if title_words else "untitled-chat"
    
    def _sanitize_filename(self, title: str) -> str:
        return re.sub(r'[^\w\-.]', '-', title)[:50]
    
    def _save_continuously(self):
        os.makedirs("chat_history", exist_ok=True)
        filepath = os.path.join("chat_history", self.filename)
        
        with open(filepath, 'w') as f:
            f.write(f"# Chat: {self.title or 'Untitled'}\n")
            f.write(f"# ID: {self.chat_id}\n")
            f.write(f"# Created: {self.messages[0]['timestamp'] if self.messages else 'Unknown'}\n\n")
            
            for msg in self.messages:
                timestamp = msg['timestamp']
                role = msg['role'].upper()
                content = msg['content']
                f.write(f"[{timestamp}] {role}: {content}\n\n")
    
    def save_explicitly(self):
        self._save_continuously()
        return os.path.join("chat_history", self.filename)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ChatHistory':
        chat = cls()
        if os.path.exists(filepath):
            # Set filename to the loaded file's name to preserve it
            chat.filename = os.path.basename(filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line in lines:
                    if line.startswith('# ID: '):
                        chat.chat_id = line.replace('# ID: ', '')
                    elif line.startswith('# Chat: '):
                        chat.title = line.replace('# Chat: ', '')
                
                pattern = r'\[(.*?)\] (USER|ASSISTANT): (.*?)(?=\n\[|\Z)'
                matches = re.findall(pattern, content, re.DOTALL)
                
                for timestamp, role, msg_content in matches:
                    chat.messages.append({
                        "role": role.lower(),
                        "content": msg_content.strip(),
                        "timestamp": timestamp
                    })
        
        return chat
    
    @classmethod
    def list_available_chats(cls) -> List[str]:
        if not os.path.exists("chat_history"):
            return []
        return [f for f in os.listdir("chat_history") if f.endswith('.txt')]

class ClaudeChatbot:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        self.client = None
        self.current_chat = ChatHistory()
        self.running = True
        self.commands = ['/save', '/new', '/list', '/resume', '/quit', '/help', '/system', '/model', '/log']
        self.system_prompt_file = "system_prompt.txt"
        
        # Model configuration with latest versions
        self.model_options = {
            # Anthropic models
            'haiku': 'claude-3-5-haiku-20241022',
            'sonnet': 'claude-sonnet-4-20250514', 
            'opus': 'claude-opus-4-20250514',
            # OpenAI models - latest versions
            'gpt4': 'gpt-4o',
            'gpt4-latest': 'chatgpt-4o-latest',
            'gpt4-mini': 'gpt-4o-mini',
            # Reasoning models
            'o1': 'o1',
            'o1-mini': 'o1-mini',
            'o1-pro': 'o1-pro',
            # Next generation models
            'o4-mini': 'o4-mini'
        }
        # Logging configuration
        self.log_levels = {
            'trace': 5,  # Custom level below DEBUG for full API tracing
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        self.config_file = "clommand_config.json"
        
        # Load configuration
        config = self._load_config()
        self.current_model = config.get('model', 'haiku')
        self.current_log_level = config.get('log_level', 'info')
        
        self._setup_logging()
        self._setup_client()
        self._setup_readline()
        self._ensure_system_prompt_exists()
    
    def _get_prompt(self) -> str:
        if self.current_chat.title:
            return f"{self.current_chat.title}::{self.current_model}> "
        else:
            return f"untitled-chat::{self.current_model}> "
    
    def _setup_readline(self):
        readline.set_completer(self._completer)
        readline.parse_and_bind('tab: complete')
        # Use default delimiters for better built-in behavior
        readline.set_completer_delims(' \t\n')
    
    def _completer(self, text: str, state: int):
        buffer = readline.get_line_buffer()
        
        # Get all possible completions for the current context
        options = self._get_completions(text, buffer)
        
        # Return the option at the requested state
        if state < len(options):
            return options[state]
        return None
    
    def _get_completions(self, text: str, buffer: str) -> List[str]:
        # If we're completing a command at the start of the line
        if buffer.startswith('/') and ' ' not in buffer:
            return [cmd for cmd in self.commands if cmd.startswith(text)]
        
        # If we're completing after "/resume "
        if buffer.startswith('/resume '):
            filename_part = buffer[8:]  # Remove "/resume "
            chat_files = ChatHistory.list_available_chats()
            
            # Filter files that start with the current text
            matching_files = [f for f in chat_files if f.startswith(filename_part)]
            
            # Return the full filename, not just the remaining part
            return matching_files
        
        # If we're completing after "/model "
        if buffer.startswith('/model '):
            model_part = buffer[7:]  # Remove "/model "
            model_names = list(self.model_options.keys())
            
            # Filter models that start with the current text
            matching_models = [m for m in model_names if m.startswith(model_part)]
            
            return matching_models
        
        # If we're completing after "/log "
        if buffer.startswith('/log '):
            log_part = buffer[5:]  # Remove "/log "
            log_levels = list(self.log_levels.keys())
            
            # Filter log levels that start with the current text
            matching_levels = [l for l in log_levels if l.startswith(log_part)]
            
            return matching_levels
        
        return []
    
    def _setup_logging(self):
        """Configure logging to write to logs/ directory"""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Add custom TRACE level
        logging.addLevelName(5, 'TRACE')
        
        # Create logger
        self.logger = logging.getLogger('clommand')
        self.logger.setLevel(self.log_levels[self.current_log_level])
        
        # Add trace method to logger
        def trace(message, *args, **kwargs):
            if self.logger.isEnabledFor(5):
                self.logger._log(5, message, args, **kwargs)
        self.logger.trace = trace
        
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        log_filename = f"logs/clommand_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(self.log_levels[self.current_log_level])
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Configure warnings to go to the logger instead of stderr
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger('py.warnings')
        
        # Remove any existing handlers from warnings logger
        for handler in warnings_logger.handlers[:]:
            warnings_logger.removeHandler(handler)
        
        # Set level and add our file handler
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.addHandler(file_handler)
        warnings_logger.propagate = False  # Prevent double logging
        
        self.logger.info(f"Logging initialized at level: {self.current_log_level}")
    
    def _ensure_system_prompt_exists(self):
        """Create default system prompt file if it doesn't exist"""
        if not os.path.exists(self.system_prompt_file):
            default_prompt = "You are Claude, an AI assistant created by Anthropic. You are helpful, harmless, and honest."
            with open(self.system_prompt_file, 'w') as f:
                f.write(default_prompt)
    
    def _get_system_prompt(self) -> str:
        """Read the system prompt from file"""
        try:
            with open(self.system_prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "You are Claude, an AI assistant created by Anthropic."
    
    def _load_config(self) -> dict:
        """Load configuration from config file, returning defaults if not found"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_config(self, updates: dict):
        """Save configuration updates to config file"""
        try:
            # Load existing config or create new one
            config = self._load_config()
            
            # Update with new values
            config.update(updates)
            
            # Save config
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Could not save config: {e}")
    
    def _show_configuration(self):
        """Display current configuration on startup"""
        # Show available providers
        providers = []
        if self.anthropic_client:
            providers.append("Anthropic")
        if self.openai_client:
            providers.append("OpenAI")
        
        provider_status = ", ".join(providers) if providers else "None"
        current_provider = self._get_model_provider(self.current_model).title()
        
        config_info = f"""Configuration:
  Model: {self.current_model} ({self.model_options[self.current_model]}) [{current_provider}]
  Log Level: {self.current_log_level}
  System Prompt: {'custom' if os.path.exists(self.system_prompt_file) else 'default'}
  Available Providers: {provider_status}"""
        
        print(config_info)
        if hasattr(self, 'logger'):
            self.logger.info(f"Started with model={self.current_model}, log_level={self.current_log_level}, providers={provider_status}")
    
    def _open_system_prompt_in_editor(self):
        """Open system prompt file in editor with proper terminal handling"""
        try:
            # Try common editors in order of preference
            editors = [
                os.environ.get('EDITOR'),
                'vim',
                'nano',
                'vi',
                'code'
            ]
            
            for editor in editors:
                if editor and self._is_command_available(editor):
                    # For vim users: execute in a way that preserves terminal state
                    if editor in ['vim', 'vi', 'nano']:
                        # Create a subprocess that inherits the terminal properly
                        process = subprocess.Popen([editor, self.system_prompt_file])
                        process.wait()  # Wait for editor to close
                        
                        # Reset terminal state quietly without clearing screen
                        os.system('stty sane')
                        
                        # Re-setup readline without fanfare
                        readline.clear_history()
                        self._setup_readline()
                        
                        print(f"System prompt updated using {editor}")
                    else:
                        # For GUI editors, just run normally
                        subprocess.run([editor, self.system_prompt_file])
                        print(f"System prompt opened in {editor}")
                    
                    return
            
            print("No suitable editor found. Please set the EDITOR environment variable.")
            print(f"You can manually edit: {self.system_prompt_file}")
            
        except Exception as e:
            print(f"Error opening editor: {e}")
            print(f"You can manually edit: {self.system_prompt_file}")
    
    def _is_command_available(self, command: str) -> bool:
        """Check if a command is available in the system PATH"""
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.run([command, '--version'], 
                             stdout=devnull, 
                             stderr=devnull, 
                             timeout=2)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _setup_client(self):
        # Initialize both clients if API keys are available
        self.anthropic_client = None
        self.openai_client = None
        
        # Setup Anthropic client
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        if anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            except Exception as e:
                print(f"Error initializing Anthropic client: {e}")
        
        # Setup OpenAI client  
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
        
        # Set primary client for backward compatibility
        self.client = self.anthropic_client or self.openai_client
        
        if not self.client:
            print("Warning: No API keys found.")
            print("Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY to use AI functionality.")
    
    def _get_model_provider(self, model_name: str) -> str:
        """Determine which provider (anthropic/openai) handles the given model"""
        anthropic_models = ['haiku', 'sonnet', 'opus']
        return 'anthropic' if model_name in anthropic_models else 'openai'
    
    def _get_ai_response(self, user_input: str) -> str:
        provider = self._get_model_provider(self.current_model)
        model = self.model_options[self.current_model]
        
        # Check if the required client is available
        client = self.anthropic_client if provider == 'anthropic' else self.openai_client
        if not client:
            error_msg = f"Error: {provider.title()} client not initialized. Please check your API key."
            self.logger.error(error_msg)
            return error_msg
        
        try:
            self.logger.debug(f"Processing user input: {user_input[:50]}...")
            
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in self.current_chat.messages]
            
            system_prompt = self._get_system_prompt()
            
            self.logger.info(f"Making {provider} API request to {model} with {len(messages)} messages")
            
            if provider == 'anthropic':
                return self._call_anthropic_api(client, model, system_prompt, messages)
            else:
                return self._call_openai_api(client, model, system_prompt, messages)
                
        except Exception as e:
            error_msg = f"Error getting AI response: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _call_anthropic_api(self, client, model: str, system_prompt: str, messages: list) -> str:
        """Call Anthropic API with proper formatting"""
        # Log full API request payload at trace level
        if self.logger.isEnabledFor(5):  # TRACE level
            request_payload = {
                "model": model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": messages
            }
            self.logger.trace(f"Anthropic API request payload:\n{json.dumps(request_payload, indent=2, ensure_ascii=False)}")
        
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        
        response_text = response.content[0].text
        self.logger.debug(f"Received Anthropic response: {response_text[:50]}...")
        
        # Log full API response at trace level
        if self.logger.isEnabledFor(5):  # TRACE level
            response_data = {
                "content": response.content[0].text,
                "model": response.model,
                "usage": response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None,
                "id": response.id if hasattr(response, 'id') else None
            }
            self.logger.trace(f"Anthropic API response:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
        
        return response_text
    
    def _emulate_system_prompt_for_o1(self, system_prompt: str, messages: list) -> list:
        """Emulate system prompt for o1 models by prepending to first user message"""
        if not messages:
            return messages
        
        # Create a copy to avoid modifying the original chat history
        openai_messages = []
        system_prompt_added = False
        
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and not system_prompt_added:
                # Check if system prompt is already embedded in this message
                # (to handle model switching scenarios)
                content = msg["content"]
                if not content.startswith(system_prompt):
                    # Prepend system prompt to first user message
                    modified_content = f"{system_prompt}\n\n{content}"
                    openai_messages.append({
                        "role": "user",
                        "content": modified_content
                    })
                else:
                    # System prompt already embedded, use as-is
                    openai_messages.append(msg.copy())
                system_prompt_added = True
            else:
                # Copy other messages as-is
                openai_messages.append(msg.copy())
        
        return openai_messages
    
    def _clean_embedded_system_prompt(self, system_prompt: str, messages: list) -> list:
        """Remove embedded system prompt from first user message for regular models"""
        if not messages:
            return messages
            
        cleaned_messages = []
        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] == "user":
                content = msg["content"]
                # Check if the message starts with the system prompt
                if content.startswith(system_prompt):
                    # Remove the system prompt and leading whitespace
                    cleaned_content = content[len(system_prompt):].lstrip('\n ')
                    if cleaned_content:  # Make sure there's still content left
                        cleaned_messages.append({
                            "role": "user", 
                            "content": cleaned_content
                        })
                    # If no content left after removing system prompt, skip this message
                else:
                    # No embedded system prompt, use as-is
                    cleaned_messages.append(msg.copy())
            else:
                cleaned_messages.append(msg.copy())
                
        return cleaned_messages
    
    def _call_openai_api(self, client, model: str, system_prompt: str, messages: list) -> str:
        """Call OpenAI API with proper formatting"""
        # o1 models don't support system messages
        if model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
            # Emulate system prompt by prepending to first user message
            openai_messages = self._emulate_system_prompt_for_o1(system_prompt, messages)
        else:
            # Clean any embedded system prompts from messages for regular models
            cleaned_messages = self._clean_embedded_system_prompt(system_prompt, messages)
            # Convert messages format for OpenAI (system message goes in messages array)
            openai_messages = [{"role": "system", "content": system_prompt}] + cleaned_messages
        
        # o1 models use different parameter name for max tokens and need more tokens for reasoning
        if model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
            token_param = "max_completion_tokens"
            max_tokens = 4096  # o1 models need more tokens for reasoning + response
        else:
            token_param = "max_tokens"
            max_tokens = 1024  # Regular models
        
        # Log full API request payload at trace level
        if self.logger.isEnabledFor(5):  # TRACE level
            request_payload = {
                "model": model,
                token_param: max_tokens,
                "messages": openai_messages
            }
            self.logger.trace(f"OpenAI API request payload:\n{json.dumps(request_payload, indent=2, ensure_ascii=False)}")
        
        # Create request parameters dynamically
        request_params = {
            "model": model,
            token_param: max_tokens,
            "messages": openai_messages
        }
        
        response = client.chat.completions.create(**request_params)
        
        response_text = response.choices[0].message.content
        
        # Detect token exhaustion in o1 models (empty response with high token usage)
        if model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
            if not response_text or response_text.strip() == "":
                usage = response.usage
                if usage and hasattr(usage, 'completion_tokens_details'):
                    reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
                    total_completion = usage.completion_tokens
                    
                    if reasoning_tokens > 0 and total_completion >= max_tokens * 0.9:  # Used 90%+ of token limit
                        error_msg = f"Error: {model} exhausted its token limit during reasoning. "
                        error_msg += f"Used {total_completion}/{max_tokens} tokens ({reasoning_tokens} for reasoning). "
                        error_msg += "Try asking a simpler question or breaking your request into smaller parts."
                        self.logger.warning(f"Token exhaustion detected: {usage.completion_tokens}/{max_tokens} tokens used")
                        return error_msg
                
                # Generic empty response error if we can't determine the cause
                return f"Error: {model} returned an empty response. This may indicate a token limit issue or other API problem."
        
        self.logger.debug(f"Received OpenAI response: {response_text[:50]}...")
        
        # Log full API response at trace level
        if self.logger.isEnabledFor(5):  # TRACE level
            response_data = {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None,
                "id": response.id if hasattr(response, 'id') else None
            }
            self.logger.trace(f"OpenAI API response:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
        
        return response_text
    
    def _format_response(self, text: str) -> str:
        """Format response text to wrap at 75 characters while preserving existing newlines"""
        # Split text into paragraphs (separated by double newlines)
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            # Split paragraph into lines (single newlines)
            lines = paragraph.split('\n')
            formatted_lines = []
            
            for line in lines:
                # Only wrap lines that are longer than 75 characters
                if len(line) <= 75:
                    formatted_lines.append(line)
                else:
                    # Wrap long lines while preserving word boundaries
                    wrapped_lines = textwrap.wrap(line, width=75, break_long_words=False, break_on_hyphens=False)
                    formatted_lines.extend(wrapped_lines)
            
            # Rejoin lines with single newlines
            formatted_paragraphs.append('\n'.join(formatted_lines))
        
        # Rejoin paragraphs with double newlines
        return '\n\n'.join(formatted_paragraphs)
    
    def _handle_command(self, command: str) -> bool:
        command = command.strip().lower()
        
        if command == '/quit' or command == '/q':
            self.logger.info("User initiated shutdown")
            self.running = False
            print("Goodbye!")
            return True
        
        elif command == '/save':
            filepath = self.current_chat.save_explicitly()
            self.logger.info(f"Chat saved to: {filepath}")
            print(f"Chat saved to: {filepath}")
            return True
        
        elif command == '/new':
            self.logger.info("Started new chat session")
            self.current_chat = ChatHistory()
            print("Started new chat session.")
            return True
        
        elif command == '/list':
            chats = ChatHistory.list_available_chats()
            if chats:
                # Get file paths with modification times
                chat_files = []
                for chat in chats:
                    filepath = os.path.join("chat_history", chat)
                    if os.path.exists(filepath):
                        mtime = os.path.getmtime(filepath)
                        chat_files.append((chat, mtime))
                
                # Sort by modification time (newest first)
                chat_files.sort(key=lambda x: x[1], reverse=True)
                
                print("Available chat histories:")
                for chat, mtime in chat_files:
                    # Format timestamp as 2025-07-12 15:33
                    formatted_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"  - {formatted_time} {chat}")
            else:
                print("No saved chat histories found.")
            return True
        
        elif command.startswith('/resume '):
            filename = command[8:].strip()
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            filepath = os.path.join("chat_history", filename)
            if os.path.exists(filepath):
                self.current_chat = ChatHistory.load_from_file(filepath)
                self.logger.info(f"Resumed chat: {self.current_chat.title} with {len(self.current_chat.messages)} messages")
                print(f"Resumed chat: {self.current_chat.title}")
                print(f"Messages loaded: {len(self.current_chat.messages)}")
                
                # Show the last turn for context
                if len(self.current_chat.messages) >= 2:
                    last_user = None
                    last_assistant = None
                    
                    # Find the last user and assistant messages
                    for msg in reversed(self.current_chat.messages):
                        if msg["role"] == "assistant" and last_assistant is None:
                            last_assistant = msg["content"]
                        elif msg["role"] == "user" and last_user is None:
                            last_user = msg["content"]
                        
                        if last_user and last_assistant:
                            break
                    
                    print("\n--- Last conversation turn ---")
                    if last_user:
                        print(f"You: {last_user}")
                    if last_assistant:
                        print(f"Claude: {last_assistant}")
                    print("--- End of context ---")
            else:
                print(f"Chat file not found: {filename}")
            return True
        
        elif command == '/system':
            self._open_system_prompt_in_editor()
            return True
        
        elif command.startswith('/model'):
            parts = command.split(' ', 1)
            if len(parts) == 1:
                # Just "/model" - show current model
                current_full_model = self.model_options[self.current_model]
                print(f"Current model: {self.current_model} ({current_full_model})")
            else:
                # "/model <name>" - switch model
                model_name = parts[1].strip().lower()
                if model_name in self.model_options:
                    # Check if required client is available
                    provider = self._get_model_provider(model_name)
                    required_client = self.anthropic_client if provider == 'anthropic' else self.openai_client
                    
                    if not required_client:
                        print(f"Error: {provider.title()} API key not configured.")
                        print(f"Please set {provider.upper()}_API_KEY to use {model_name} model.")
                        return True
                    
                    self.current_model = model_name
                    current_full_model = self.model_options[self.current_model]
                    self._save_config({'model': model_name})  # Persist the setting
                    self.logger.info(f"Switched to model: {self.current_model} ({current_full_model})")
                    print(f"Switched to model: {self.current_model} ({current_full_model})")
                else:
                    self.logger.warning(f"Invalid model requested: {model_name}")
                    print(f"Unknown model: {model_name}")
                    print(f"Available models: {', '.join(self.model_options.keys())}")
            return True
        
        elif command.startswith('/log'):
            parts = command.split(' ', 1)
            if len(parts) == 1:
                # Just "/log" - show current log level
                print(f"Current log level: {self.current_log_level}")
                print(f"Available levels: {', '.join(self.log_levels.keys())}")
            else:
                # "/log <level>" - switch log level
                log_level = parts[1].strip().lower()
                if log_level in self.log_levels:
                    self.current_log_level = log_level
                    self._setup_logging()  # Reconfigure logging with new level
                    self._save_config({'log_level': log_level})  # Persist the setting
                    print(f"Log level set to: {self.current_log_level}")
                    self.logger.info(f"Log level changed to: {self.current_log_level}")
                else:
                    print(f"Unknown log level: {log_level}")
                    print(f"Available levels: {', '.join(self.log_levels.keys())}")
            return True
        
        elif command == '/help':
            print("Available commands:")
            print("  /save     - Save current chat history")
            print("  /new      - Start a new chat session")
            print("  /list     - List available chat histories")
            print("  /resume <filename> - Resume a previous chat")
            print("  /system   - Edit system prompt")
            print("  /model    - Show current model")
            print(f"  /model <name> - Switch model ({', '.join(self.model_options.keys())})")
            print("  /log      - Show current log level")
            print("  /log <level> - Set log level (trace, debug, info, warning, error)")
            print("  /quit     - Exit the chatbot")
            print("  /help     - Show this help message")
            return True
        
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.")
            return True
    
    def _get_input_with_paste_detection(self, prompt: str) -> str:
        """Get user input with bracket paste mode detection."""
        import sys
        import tty
        import termios
        import select
        
        # Check if stdin is a terminal
        if not sys.stdin.isatty():
            return input(prompt)
        
        print(prompt, end='', flush=True)
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for character-by-character input
            tty.setraw(sys.stdin.fileno())
            
            input_buffer = []
            in_paste_mode = False
            
            while True:
                # Check if data is available
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    char = sys.stdin.read(1)
                    
                    # Check for bracket paste start sequence (\033[200~)
                    if len(input_buffer) >= 5 and input_buffer[-5:] == ['\033', '[', '2', '0', '0'] and char == '~':
                        in_paste_mode = True
                        input_buffer = input_buffer[:-5]  # Remove the escape sequence
                        continue
                    
                    # Check for bracket paste end sequence (\033[201~)
                    if in_paste_mode and len(input_buffer) >= 5 and input_buffer[-5:] == ['\033', '[', '2', '0', '1'] and char == '~':
                        in_paste_mode = False
                        input_buffer = input_buffer[:-5]  # Remove the escape sequence
                        continue
                    
                    # Handle Enter key
                    if char == '\r' or char == '\n':
                        if not in_paste_mode:
                            # Normal Enter - submit input
                            print()  # Move to next line
                            break
                        else:
                            # Pasted newline - add literal newline
                            input_buffer.append(char)
                            print(char, end='', flush=True)
                    
                    # Handle backspace
                    elif char == '\x7f' or char == '\x08':
                        if input_buffer:
                            input_buffer.pop()
                            print('\b \b', end='', flush=True)
                    
                    # Handle Ctrl+C
                    elif char == '\x03':
                        raise KeyboardInterrupt
                    
                    # Handle Ctrl+D
                    elif char == '\x04':
                        if not input_buffer:
                            raise EOFError
                    
                    # Regular character
                    else:
                        input_buffer.append(char)
                        if char == '\n' and in_paste_mode:
                            print('\n', end='', flush=True)
                        else:
                            print(char, end='', flush=True)
                
                else:
                    # No input available, sleep briefly
                    import time
                    time.sleep(0.01)
            
            return ''.join(input_buffer)
            
        except (termios.error, tty.error):
            # Fallback to regular input if terminal manipulation fails
            return input(prompt)
            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def run(self):
        print("Claude Command-Line Chatbot")
        print("Type /help for commands, or start chatting!")
        print("=" * 50)
        
        # Show current configuration
        self._show_configuration()
        print("=" * 50)
        
        while self.running:
            try:
                user_input = self._get_input_with_paste_detection(f"\n{self._get_prompt()}")
                
                if user_input.startswith('/'):
                    if user_input.strip() == '/':
                        print("Available commands:")
                        print("  /save     - Save current chat history")
                        print("  /new      - Start a new chat session")
                        print("  /list     - List available chat histories")
                        print("  /resume <filename> - Resume a previous chat")
                        print("  /system   - Edit system prompt")
                        print("  /model    - Show current model")
                        print(f"  /model <name> - Switch model ({', '.join(self.model_options.keys())})")
                        print("  /log      - Show current log level")
                        print("  /log <level> - Set log level (trace, debug, info, warning, error)")
                        print("  /quit     - Exit the chatbot")
                        print("  /help     - Show this help message")
                    else:
                        self._handle_command(user_input.strip())
                    continue
                
                user_input = user_input.strip()
                if not user_input:
                    continue
                
                # Add user message to history first (needed for API context)
                self.current_chat.add_message("user", user_input)
                
                response = self._get_ai_response(user_input)
                
                # Check if response is an error (starts with "Error:")
                if response.startswith("Error:"):
                    # Remove the user message from history since exchange failed
                    self.current_chat.messages.pop()
                    # Don't add error to chat history, just display it
                    print(response)
                else:
                    # Normal response - add assistant response to chat history
                    formatted_response = self._format_response(response)
                    print(formatted_response)
                    self.current_chat.add_message("assistant", response)
                
            except KeyboardInterrupt:
                print("\n\nUse /quit to exit gracefully.")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    chatbot = ClaudeChatbot()
    chatbot.run()