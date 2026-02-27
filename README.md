---
title: AI Research Assistant
emoji: 📑
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.31.0"
app_file: app.py
pinned: false
---

# AI Research Assistant

An intelligent PDF analysis tool powered by AI. Upload any PDF document and ask questions about it!

## Features

- 📄 **PDF Text Extraction** - Automatically extracts and processes text from PDFs
- 🧠 **Semantic Search** - Uses AI embeddings to understand context and meaning
- 🤖 **AI-Powered Answers** - Generates natural language answers using Google Gemini
- 💬 **Chat Interface** - Interactive chat-style Q&A
- 📚 **Source Citations** - Shows which parts of the document were used for answers

## How to Use

1. Upload a PDF document using the sidebar
2. Wait for processing to complete (usually 30-60 seconds)
3. Ask questions about your document in the chat
4. Get intelligent answers based on the document's content!

## Tech Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS
- **LLM**: Google Gemini 1.5 Flash

## Example Questions

- What is the main conclusion?
- What methodology was used?
- What are the key findings?
- Who are the authors?
- What datasets were mentioned?
