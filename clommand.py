#!/usr/bin/env python3

import os
import sys
import json
import re
import hashlib
import readline
from datetime import datetime
from typing import List, Dict, Optional
import anthropic

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
                self.filename = f"{self._sanitize_filename(self.title)}.txt"
                if os.path.exists(old_filename) and old_filename != self.filename:
                    os.rename(old_filename, self.filename)
    
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
        self.client = None
        self.current_chat = ChatHistory()
        self.running = True
        self.commands = ['/save', '/new', '/list', '/resume', '/quit', '/help']
        self._setup_client()
        self._setup_readline()
    
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
        
        return []
    
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
            return "Error: Claude client not initialized. Please check your API key."
        
        try:
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in self.current_chat.messages]
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages
            )
            
            return response.content[0].text
        except Exception as e:
            return f"Error getting Claude response: {e}"
    
    
    def _handle_command(self, command: str) -> bool:
        command = command.strip().lower()
        
        if command == '/quit' or command == '/q':
            self.running = False
            print("Goodbye!")
            return True
        
        elif command == '/save':
            filepath = self.current_chat.save_explicitly()
            print(f"Chat saved to: {filepath}")
            return True
        
        elif command == '/new':
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
        
        elif command == '/help':
            print("Available commands:")
            print("  /save     - Save current chat history")
            print("  /new      - Start a new chat session")
            print("  /list     - List available chat histories")
            print("  /resume <filename> - Resume a previous chat")
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