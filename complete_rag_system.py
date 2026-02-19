from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from embeddings import load_embedding_model, get_embeddings
from vector_store import VectorStore
from llm_engine import generate_answer
import os
from dotenv import load_dotenv

API_KEY = os.getenv("GEMINI_API_KEY")  

def run_rag_system():
    """Complete RAG system - PDF to Answer"""
    
    print("="*60)
    print("AI RESEARCH ASSISTANT")
    print("="*60)
    
    #loading pdf
    pdf_path = input("\nEnter path to your PDF: ")
    print("\n Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print(" Failed to extract text!")
        return
    
    print(f"✓ Extracted {len(text)} characters")
    
    #chunking text
    print("\n Step 2: Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"✓ Created {len(chunks)} chunks")
    
    #creating embeddings
    print("\n Step 3: Creating embeddings...")
    model = load_embedding_model()
    embeddings = get_embeddings(chunks, model)
    print(f"✓ Created {len(embeddings)} embeddings")
    
    #building vector store
    print("\n Step 4: Building vector store...")
    store = VectorStore()
    store.add_documents(chunks, embeddings)
    
    #interactive q/a
    print("\n" + "="*60)
    print(" SYSTEM READY! Ask questions about your PDF!")
    print("="*60)
    print("(Type 'quit' to exit)\n")
    
    while True:
        query = input(" Your question: ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n Thanks for using the AI Research Assistant!")
            break
        
        if not query.strip():
            continue
        
        #finding relevant chunks
        print("\n Searching for relevant information...")
        query_emb = model.encode(query)
        indices, distances = store.search(query_emb, top_k=3)
        relevant_chunks = [chunks[i] for i in indices]
        
        #generate answer
        print(" Generating answer...\n")
        answer = generate_answer(query, relevant_chunks, API_KEY)
        
        print(" Answer:")
        print(f"{answer}\n")
        print("-"*60)

if __name__ == "__main__":
    run_rag_system()