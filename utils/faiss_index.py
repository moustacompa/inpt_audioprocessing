import faiss
import numpy as np
import os
def getIndex():
    embeddings = np.load(os.path.join("embeddings","text_embeddings.npy"))
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print("Documents indexés:", index.ntotal)
    return index