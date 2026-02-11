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
        self.chunks.extend(chunks)                                         #faiss only stores numbers, not text
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