# AI Document Q&A RAG App

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions based on the uploaded content. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and generate context-aware answers using an LLM.

---

## Project Overview

This project solves a common problem: users often need to quickly understand large PDF documents without manually reading every page.

Instead of asking a general chatbot, this application answers questions using the actual uploaded documents as context. This helps reduce hallucination and makes the answers more relevant to the provided content.

---

## Key Features

- Upload multiple PDF documents
- Ask questions based on uploaded files
- Retrieve relevant document chunks using semantic search
- Generate answers using an LLM
- Display source documents used for the answer
- Simple Streamlit interface for quick testing and demo

---

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- HuggingFace Embeddings
- Unstructured Document Loader
- Recursive Character Text Splitter

---

## How It Works

1. The user uploads one or more PDF files.
2. The application saves the uploaded files locally.
3. The documents are loaded and split into smaller chunks.
4. HuggingFace embeddings are created for each chunk.
5. The chunks are stored in a Chroma vector database.
6. When the user asks a question, the app retrieves the most relevant chunks.
7. The LLM generates an answer using the retrieved context.
8. The app displays both the answer and the source documents.

---

## Why This Project Matters

Large Language Models are powerful, but they can hallucinate when they do not have access to the right information. This project demonstrates how RAG can make AI applications more reliable by grounding responses in actual documents.

This type of system is useful for:

- Research document analysis
- Business reports
- Legal or policy documents
- Academic papers
- Internal company knowledge bases
- Customer support documentation

---

## Folder Structure

```text
RAG-QA-APP/
│
├── app.py              # Streamlit frontend and user interaction
├── rag_utility.py      # RAG pipeline, embeddings, vector DB, and QA chain
├── requirements.txt    # Python dependencies
├── env.txt             # Environment variable reference
└── .gitignore
