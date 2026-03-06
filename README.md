# Chatbot with Long-Term Memory (sqlite-vec + RAG)

## The problem

LLMs don’t update their weights after training. In practice, that means they can’t keep new information across sessions unless it is provided again in the prompt/context. A common workaround is to persist memories externally in a file or database and inject them in the contex of the new conversation in some way.

This project is an attempt at one such way.

## This chatbot with memory

This is a normal chatbot, but it can store user-related information in a database, such as “User likes sports” or “User’s name is John”. Memories are separated into two categories: important and less relevant.

There is a fixed maximum ammount (say N) of important memories, and they are always included in the prompt. If a new memory is considered more important than the least-important memory currently in the important set, it replaces it. The displaced memory is then demoted into the less relevant pool. The heuristic/proxy for calculating importance is an aggregate similarity score against a few “anchor” texts designed to try to capture different aspects of relevance (long-term stability, interaction utility, and “not noise”).

Individual less relevant memories are not always included. Instead, on each user message, the message is embedded and a RAG step retrieves the top-K most similar memories from the less relevant pool. Those retrieved memories are then injected into the prompt. This keeps memories contextual and responses better tuned to the topic at hand, mimicking human associative memory.

### Context compaction

To keep latency low and avoid filling the context window, the conversation is periodically summarized. The system keeps a general summary of older conversation, a summary of the most recent summarized batch, and the most recent raw messages so the model never loses live conversational continuity.

## Tools used

Python, LangChain, OpenAI Embeddings, uv package manager, sqlite/sqlite-vec.

## How to run

1. Clone the repo;
1. In the terminal, in the project directory, type `uv sync` to install the dependencies (you need to have the uv package manager installed);
1. Create a file named ".env", and add in it the line `OPENAI_API_KEY = your-OpenAI-API-key` (replace "your-OpenAI-API-key" with your actual OpenAI API key);
1. In the terminal, in the project directory, type `uv run ./src/main.py` to start the chat;
1. To leave the chat and exit the program, type "/exit" in the chat.
