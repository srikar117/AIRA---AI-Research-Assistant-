from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

def load_embedding_model():
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model

def get_embeddings(texts, model):
    embeddings = model.encode(texts)
    return embeddings

def find_most_similar(query, chunks, chunk_embeddings, model, top_k = 5):
    query_embedding = model.encode(query)
    similarities = cos_sim(query_embedding, chunk_embeddings)
    similarity_scores = similarities[0].cpu().numpy()      #converts pytorch tensor to numpy array and runs it on cpu instead of gpu
                                                           #numpy only works with cpu data
    print("\nDEBUG - Similarity scores:")
    for i, score in enumerate(similarity_scores):
        print(f"  Chunk {i}: {score:.4f} - {chunks[i][:50]}...")

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    print(f"\nTop indices: {top_indices}")  
    print(f"Top scores: {similarity_scores[top_indices]}")
    return top_indices

#testing
if __name__ == "__main__":
    chunks = [
        "The Eiffel Tower is in Paris, France.",
        "Python is a programming language.",
        "The capital of France is Paris.",
        "Machine learning uses neural networks.",
        "Paris is known for its beautiful architecture."
    ]
    
    print("Loading model...")
    model = load_embedding_model()
    print("✓ Model loaded!\n")
    
    print("Generating embeddings...")
    chunk_embeddings = get_embeddings(chunks, model)
    print(f"✓ Generated {len(chunk_embeddings)} embeddings")
    print(f"  Each embedding has {len(chunk_embeddings[0])} dimensions\n")
    
    query = "What is the capital of France?"
    print(f"Query: '{query}'\n")
    
    similar_indices = find_most_similar(query, chunks, chunk_embeddings, model, top_k=3)
    
    print("Most similar chunks:")
    for i, idx in enumerate(similar_indices, 1):
        print(f"{i}. {chunks[idx]}")