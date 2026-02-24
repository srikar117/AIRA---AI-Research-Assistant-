import streamlit as st
import os
from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from embeddings import load_embedding_model, get_embeddings
from vector_store import VectorStore
from llm_engine import generate_answer
from dotenv import load_dotenv
import tempfile

#page configuration
st.set_page_config(
    page_title = "AIRA",
    page_icon = "📑",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

load_dotenv()
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Please set an api key")
    st.stop()

#initializing session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'chunks' not in st.session_state:
    st.session_state.chunks = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

#header
st.title("AI Research Assistant")
st.markdown(""" 
Upload a PDF document and ask questions about it. The AI will analyze the document 
and provide answers based on its content.
""")

#sidebar for uploading file
with st.sidebar:
    st.header("Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type = ['pdf'],
        help = "Upload a research paper, article, ar any PDF document"
    )

    if uploaded_file is not None:
        #check if the file is new
        if st.session_state.uploaded_filename != uploaded_file.name:
            #reset state for new file
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.session_state.uploaded_filename = uploaded_file.name
        
        if not st.session_state.processed:
            with st.spinner("🔄 Processing PDF... This may take a minute..."):
                try:
                    #saving uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    #extracting text
                    st.sidebar.info("📄 Extracting text from PDF...")
                    text = extract_text_from_pdf(tmp_path)
                    
                    if not text or len(text.strip()) < 100:
                        st.sidebar.error("Could not extract sufficient text from PDF. Please try another file.")
                        st.stop()
                    
                    #chunking text
                    st.sidebar.info("✂️ Chunking text...")
                    chunks = chunk_text(text, chunk_size=500, overlap=100)
                    
                    #loading embedding model
                    if st.session_state.model is None:
                        st.sidebar.info("🧠 Loading AI model...")
                        st.session_state.model = load_embedding_model()
                    
                    #creating embeddings
                    st.sidebar.info("🔢 Creating embeddings...")
                    embeddings = get_embeddings(chunks, st.session_state.model)
                    
                    #building vector store
                    st.sidebar.info("🗄️ Building vector database...")
                    store = VectorStore()
                    store.add_documents(chunks, embeddings)
                    
                    #save to session state
                    st.session_state.vector_store = store
                    st.session_state.chunks = chunks
                    st.session_state.processed = True
                    
                    #clean up temp file
                    os.unlink(tmp_path)
                    
                    st.sidebar.success(f"Successfully processed!\n\n📊 {len(chunks)} chunks created")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing PDF: {str(e)}")
                    st.stop()
    
    #show document info if processed
    if st.session_state.processed:
        st.markdown("---")
        st.markdown("### Current Document")
        st.info(f"**{st.session_state.uploaded_filename}**")
        st.markdown(f" **Chunks:** {len(st.session_state.chunks)}")
        
        #clear button
        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.session_state.vector_store = None
            st.session_state.chunks = None
            st.session_state.uploaded_filename = None
            st.rerun()
    
    #about section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI Research Assistant uses:
    - 📄 PDF text extraction
    - 🧠 Semantic embeddings
    - 🔍 Vector similarity search
    - 🤖 Google Gemini AI
    
    Built with Streamlit & Python
    """)

#main chat interface
#show instructions if no PDF uploaded
if not st.session_state.processed:
    st.info("👈 Upload a PDF document from the sidebar to get started!")
    
    #example questions for users
    st.markdown("### Example Questions You Can Ask:")
    st.markdown("""
    - What is the main conclusion of this paper?
    - What methodology was used in the experiments?
    - What are the key findings?
    - Who are the authors?
    - What datasets were used?
    - What are the limitations mentioned?
    """)
    
else:
    #display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        #add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        #display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        #generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    #search for relevant chunks
                    query_emb = st.session_state.model.encode(prompt)
                    indices, distances = st.session_state.vector_store.search(query_emb, top_k=3)
                    relevant_chunks = [st.session_state.chunks[i] for i in indices]
                    
                    #generate answer
                    answer = generate_answer(prompt, relevant_chunks, API_KEY)
                    
                    #display answer
                    st.markdown(answer)
                    
                    with st.expander("📚 View Sources"):
                        for i, (idx, dist) in enumerate(zip(indices, distances), 1):
                            st.markdown(f"**Source {i}** (Relevance: {1/(1+dist):.2f})")
                            st.text(st.session_state.chunks[idx][:300] + "...")
                            st.markdown("---")
                    
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
                    st.error(answer)
        
        # add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

#footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Made with ❤️ by Sai Srikar | AIRA powered by Google Gemini
</div>
""", unsafe_allow_html=True)