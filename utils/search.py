from sentence_transformers import SentenceTransformer 
import faiss_index as fi

def search(query) :
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    index = fi.getIndex()
    distances, indices = index.search(query_embedding, 5)
    return (indices, distances)