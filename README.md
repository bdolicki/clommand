# Claude Command-Line Chatbot

A command-line chatbot that integrates with Claude AI and OpenAI models, automatically managing chat history.

## Features

- Interactive command-line interface
- Support for both Anthropic (Claude) and OpenAI models
- Multiple model options including GPT-4 variants and reasoning models (o1, o1-pro)
- Automatic chat history persistence
- Smart conversation naming based on content
- Resume previous conversations
- Built-in commands for chat management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your API keys:
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
```

**Note**: You need at least one API key to use the chatbot. Both providers are optional, but having both gives you access to all available models. You can put them in an `.env` file.

3. Run the chatbot:
```bash
python clommand.py
```

## Commands

- `/save` - Explicitly save current chat history
- `/new` - Start a new chat session
- `/list` - List available chat histories
- `/resume <filename>` - Resume a previous chat
- `/model` - Show current model
- `/model <name>` - Switch between available models
- `/system` - Edit system prompt
- `/log <level>` - Set logging level (trace, debug, info, warning, error)
- `/quit` - Exit the chatbot
- `/help` - Show help message

## Available Models

**Anthropic (Claude):**
- `haiku` - Claude 3.5 Haiku (fast, efficient)
- `sonnet` - Claude Sonnet 4 (balanced)
- `opus` - Claude Opus 4 (most capable)

**OpenAI:**
- `gpt4` - GPT-4o (latest standard model)
- `gpt4-latest` - ChatGPT-4o latest version
- `gpt4-mini` - GPT-4o mini (fast, cost-effective)
- `o1` - O1 reasoning model
- `o1-mini` - O1 mini reasoning model
- `o1-pro` - O1 pro reasoning model
- `o4-mini` - O4 mini (next generation)

## Chat History

- Conversations are automatically saved to `chat_history/` directory
- Files are named based on conversation content
- Each chat has a unique ID and timestamp
- You can resume any previous conversation
