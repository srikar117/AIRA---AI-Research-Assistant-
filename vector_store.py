import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.dimension = dimension
        print(f"Created FAISS index with dimension {dimension}")

    def add_documents(self, chunks, embeddings):
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} documents (total : {len(self.chunks)})")