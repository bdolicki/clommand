# Claude Command-Line Chatbot

A command-line chatbot that lets you switch between Anthropic and OpenAI models mid-conversation.

## Example Usage
```
% ./clommand.py
Claude Command-Line Chatbot
Type /help for commands, or start chatting!
==================================================
Configuration:
  Model: haiku (claude-3-5-haiku-20241022) [Anthropic]
  Log Level: trace
  System Prompt: brief2
  Available Providers: Anthropic, OpenAI
==================================================
untitled-chat::haiku> Why is life, the Universe, and everything?
42. (A reference to Douglas Adams' humorous science fiction)
why-life-the-universe-and::haiku> /model o4-mini
why-life-the-universe-and::o4-mini> Try again
42
why-life-the-universe-and::o4-mini>
```

## Features

- Interactive command-line interface
- Support for Anthropic (Claude) and OpenAI models
- Automatic chat history persistence in individual text files named based on content
- Resume previous conversations with `/resume`
- Switch models with `/model` mid-conversation while keeping the context
- Change system prompt anytime by typing `/system`

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
