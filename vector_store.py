import faiss
import numpy as np
import pickle

class VectorStore:                                       #initialize vector store
    def __init__(self, dimension = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.dimension = dimension
        print(f"Created FAISS index with dimension {dimension}")

    def add_documents(self, chunks, embeddings):          #add chunks and their embeddings to the store
        embeddings_array = np.array(embeddings).astype('float32')          #faiss only takes this type
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)                                         #faiss only stores numbers, chunk text is stored in the python list
        print(f"Added {len(chunks)} documents (total : {len(self.chunks)})")

    def search(self, query_embedding, top_k = 5):         #search for similar documents    
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0], distances[0]
    
    def save(self, filepath):                             #save the index and chunks to disk
        faiss.write_index(self.index, f"{filepath}.index")

        with open(f"{filepath}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"Saved to {filepath}.index and {filepath}.chunks")

    def load(self, filepath):                             #load index and chunks from the disk
        self.index = faiss.read_index(f"{filepath}.index")

        with open(f"{filepath}.chunks", 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"Loaded {len(self.chunks)} chunks form {filepath}")

#testing
if __name__ == "__main__":
    from pdf_processor import extract_text_from_pdf
    from text_chunker import chunk_text
    from embeddings import load_embedding_model, get_embeddings
    
    # 1. Extract text from a real PDF
    pdf_path = input("Enter path to a PDF file: ")
    print("\n1. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        # 2. Chunk the text
        print("2. Chunking text...")
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        print(f"   Created {len(chunks)} chunks")
        
        # 3. Load model and create embeddings
        print("3. Loading model and creating embeddings...")
        model = load_embedding_model()
        embeddings = get_embeddings(chunks, model)
        print(f"   Created {len(embeddings)} embeddings")
        
        # 4. Create vector store
        print("4. Creating vector store...")
        store = VectorStore()
        store.add_documents(chunks, embeddings)
        
        # 5. Interactive search
        print("\n" + "="*50)
        print("READY! Ask questions about your PDF!")
        print("="*50)
        
        while True:
            query = input("\nYour question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            query_emb = model.encode(query)
            indices, distances = store.search(query_emb, top_k=3)
            
            print(f"\nTop 3 relevant chunks:")
            for i, (idx, dist) in enumerate(zip(indices, distances), 1):
                print(f"\n{i}. [Relevance: {1/(1+dist):.2f}]")
                print(f"   {chunks[idx][:200]}...")