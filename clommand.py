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
from datetime import datetime
from typing import List, Dict, Optional
import anthropic
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

class ClaudeREPL:
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
            'haiku': 'claude-3-5-haiku-20241022',
            'sonnet': 'claude-sonnet-4-20250514',
            'opus': 'claude-opus-4-20250514'
        }
        self.current_model = 'haiku'  # Default to haiku
        
        # Logging configuration
        self.log_levels = {
            'trace': 5,  # Custom level below DEBUG for full API tracing
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        self.current_log_level = 'info'  # Default to info
        
        self._setup_logging()
        self._setup_client()
        self._setup_readline()
        self._ensure_system_prompt_exists()
    
    def _get_prompt(self) -> str:
        if self.current_chat.title:
            return f"{self.current_chat.title}> "
        else:
            return "untitled-chat> "
    
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
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
            print("Please set it to use Claude API functionality.")
            return
        
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            print(f"Error initializing Claude client: {e}")
    
    def _get_claude_response(self, user_input: str) -> str:
        if not self.client:
            error_msg = "Error: Claude client not initialized. Please check your API key."
            self.logger.error(error_msg)
            return error_msg
        
        try:
            self.logger.debug(f"Processing user input: {user_input[:50]}...")
            
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in self.current_chat.messages]
            
            system_prompt = self._get_system_prompt()
            model = self.model_options[self.current_model]
            
            self.logger.info(f"Making API request to {model} with {len(messages)} messages")
            
            # Log full API request payload at trace level
            if self.logger.isEnabledFor(5):  # TRACE level
                request_payload = {
                    "model": model,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": messages
                }
                self.logger.trace(f"Full API request payload:\n{json.dumps(request_payload, indent=2, ensure_ascii=False)}")
            
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )
            
            response_text = response.content[0].text
            self.logger.debug(f"Received response: {response_text[:50]}...")
            
            # Log full API response at trace level
            if self.logger.isEnabledFor(5):  # TRACE level
                response_data = {
                    "content": response.content[0].text,
                    "model": response.model,
                    "usage": response.usage.dict() if hasattr(response, 'usage') and response.usage else None,
                    "id": response.id if hasattr(response, 'id') else None
                }
                self.logger.trace(f"Full API response:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
            
            return response_text
        except Exception as e:
            error_msg = f"Error getting Claude response: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    
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
                print("Available chat histories:")
                for chat in chats:
                    print(f"  - {chat}")
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
                    self.current_model = model_name
                    current_full_model = self.model_options[self.current_model]
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
            print("  /model <name> - Switch model (haiku, sonnet, opus)")
            print("  /log      - Show current log level")
            print("  /log <level> - Set log level (trace, debug, info, warning, error)")
            print("  /quit     - Exit the REPL")
            print("  /help     - Show this help message")
            return True
        
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.")
            return True
    
    def run(self):
        print("Claude REPL - Command Line Chatbot")
        print("Type /help for commands, or start chatting!")
        print("=" * 50)
        
        while self.running:
            try:
                user_input = input(f"\n{self._get_prompt()}")
                
                if user_input.startswith('/'):
                    if user_input.strip() == '/':
                        print("Available commands:")
                        print("  /save     - Save current chat history")
                        print("  /new      - Start a new chat session")
                        print("  /list     - List available chat histories")
                        print("  /resume <filename> - Resume a previous chat")
                        print("  /system   - Edit system prompt")
                        print("  /model    - Show current model")
                        print("  /model <name> - Switch model (haiku, sonnet, opus)")
                        print("  /log      - Show current log level")
                        print("  /log <level> - Set log level (trace, debug, info, warning, error)")
                        print("  /quit     - Exit the REPL")
                        print("  /help     - Show this help message")
                    else:
                        self._handle_command(user_input.strip())
                    continue
                
                user_input = user_input.strip()
                if not user_input:
                    continue
                
                self.current_chat.add_message("user", user_input)
                
                print("\nClaude:", end=" ")
                response = self._get_claude_response(user_input)
                print(response)
                
                self.current_chat.add_message("assistant", response)
                
            except KeyboardInterrupt:
                print("\n\nUse /quit to exit gracefully.")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    repl = ClaudeREPL()
    repl.run()