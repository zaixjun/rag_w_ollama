# KnowledgeBase

The `KnowledgeBase` class is designed to load documents, split them into smaller chunks, store them in a Chroma database, and answer queries using a retrieval-augmented generation (RAG) approach. This allows you to query the documents and retrieve relevant information using a prompt-based model.

## Features

- **Document Loading**: Loads documents from a specified directory.
- **Document Splitting**: Splits large documents into smaller chunks for more efficient searching and embedding.
- **Chroma Integration**: Stores the document chunks in a Chroma database for fast similarity-based search.
- **Query Answering**: Uses a model to answer questions based on the retrieved document chunks.

## Requirements

To use this class, ensure you have the following Python libraries installed:

```bash
pip install langchain langchain_chroma langchain_community ollama
```

This project is inspired by and follows concepts from the [pixegami/rag-tutorial-v2](https://github.com/pixegami/rag-tutorial-v2/tree/main) repository.