# smart-memory-chat

A terminal chatbot built with [LangGraph](https://github.com/langchain-ai/langgraph). It keeps each conversation in SQLite so you can pick it up later, and it shortens long histories by keeping only the messages that matter.

## Features

- **Saved conversations.** Every chat has a thread ID, and its state is stored in a SQLite database (`history.db` by default) through LangGraph's `SqliteSaver`. Pass the same thread ID later to continue where you left off.
- **Smart memory filtering.** When a conversation grows past 8 messages, a classifier from `langchain-typesafe` scores each message. Messages that hold important facts about the user, or that matter for following the conversation, are kept. Of the rest, only the 4 most recent stay.
- **Any OpenAI chat model.** You set the model and temperature through environment variables.

## How it works

```
START ──▶ chatbot ──(more than 8 messages?)──▶ filter
                 │
                 └──(otherwise)──▶ END
```

| Node | What it does |
| --- | --- |
| `chatbot` | Adds a system prompt to the history, calls the LLM and appends the reply. |
| `filter` | Asks the classifier about every message except the first and last. It keeps the first message, the important ones, the 4 most recent regular ones and the last message, and drops the rest. |

The `messages` state uses LangGraph's `add_messages` reducer. Normally a node's output is appended to the history. The `filter` node replaces the whole history instead: it returns `RemoveMessage(id=REMOVE_ALL_MESSAGES)` followed by the messages to keep.

### Project layout

```
src/smart_memory_chat/
├── __init__.py   # CLI entry point (argument parsing, wiring)
├── graph.py      # ChatBotGraph: state, nodes and graph definition
├── session.py    # ChatSession: interactive REPL on top of the compiled graph
└── settings.py   # Pydantic settings loaded from .env
```

## Requirements

- Python 3.15.2 or newer (see `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- A Type-Safe API key, used by the message classifier

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your keys
```

### Configuration

Settings are read from `.env` or from environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `AI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `AI_MODEL_TEMPERATURE` | `0.0` | Sampling temperature |
| `TYPE_SAFE_API_KEY` | — | API key for the `langchain-typesafe` classifier |
| `TYPE_SAFE_MODEL` | `jev-latest` | Classifier model |
| `DATABASE_NAME` | `history.db` | Path to the SQLite file where conversations are saved |

## Usage

Start a new conversation:

```bash
uv run smart-memory-chat
```

The app prints a thread ID when it starts. To continue that conversation later, pass the ID back in:

```bash
uv run smart-memory-chat --thread-id <thread-id>
```

Type `exit` or `quit`, or press `Ctrl+D` / `Ctrl+C`, to leave.

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format (2-space indentation)
uv run mypy src          # type check
```
