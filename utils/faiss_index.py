"""
faiss_index.py — Indexation FAISS des embeddings texte du corpus.

Correction P3 : IndexFlatL2 remplacé par IndexFlatIP.
Les embeddings sont normalisés L2 (par SpeechEncoder et DualEncoderModel),
donc le produit scalaire (Inner Product) équivaut à la similarité cosinus.
IndexFlatL2 sur des vecteurs normalisés est redondant et moins intuitif.
"""
import os

import faiss
import numpy as np

EMBEDDINGS_PATH = os.path.join("embeddings", "text_embeddings.npy")


def getIndex(embeddings_path: str = EMBEDDINGS_PATH) -> faiss.Index:
    """Construit et retourne un index FAISS sur les embeddings texte du corpus.

    Utilise IndexFlatIP (produit scalaire = cosine sur vecteurs normalisés L2).
    Compatible avec les embeddings produits par SpeechEncoder et DualEncoderModel.

    Paramètres
    ----------
    embeddings_path : str
        Chemin vers le fichier .npy contenant les embeddings texte.

    Retourne
    --------
    faiss.Index prêt pour la recherche top-k.
    """
    embeddings = np.load(embeddings_path).astype("float32")

    # Normaliser L2 pour que le produit scalaire == similarité cosinus
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    dimension = embeddings.shape[1]
    # CORRECTION: IndexFlatIP au lieu de IndexFlatL2
    # Sur des vecteurs normalisés, IP = cosine similarity → scores dans [-1, 1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Documents indexés : {index.ntotal} | dim={dimension}")
    return index


def save_index(index: faiss.Index, path: str) -> None:
    """Sauvegarde l'index FAISS sur disque."""
    faiss.write_index(index, path)
    print(f"Index sauvegardé → {path}")


def load_index(path: str) -> faiss.Index:
    """Charge un index FAISS depuis le disque."""
    return faiss.read_index(path)
