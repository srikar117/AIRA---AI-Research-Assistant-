import streamlit as st
import os
from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from embeddings import load_embedding_model, get_embeddings
from vector_store import VectorStore
from llm_engine import generate_answer
from dotenv import load_dotenv

st.set_page_config(
    page_title = "AIRA",
    page_icon = "📑",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

