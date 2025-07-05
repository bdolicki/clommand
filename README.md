# Claude REPL Chatbot

A command-line REPL chatbot that integrates with Claude AI and automatically manages chat history.

## Features

- Interactive command-line interface
- Automatic chat history persistence
- Smart conversation naming based on content
- Resume previous conversations
- Built-in commands for chat management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

3. Run the chatbot:
```bash
python clommand.py
```

## Commands

- `/save` - Explicitly save current chat history
- `/new` - Start a new chat session
- `/list` - List available chat histories
- `/resume <filename>` - Resume a previous chat
- `/quit` - Exit the REPL
- `/help` - Show help message

## Chat History

- Conversations are automatically saved to `chat_history/` directory
- Files are named based on conversation content
- Each chat has a unique ID and timestamp
- You can resume any previous conversation