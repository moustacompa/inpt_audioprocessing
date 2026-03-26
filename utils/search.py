"""
search.py — Moteur de recherche Speech-to-Retrieval (S2R).

Deux modes de recherche :
  - search_by_audio()  : requête AUDIO brute → top-k documents  [S2R réel, sans ASR]
  - search_by_text()   : requête TEXTE → top-k documents         [prototype texte-texte]

Le mode audio est l'objectif du projet : convertir la parole en embedding
via Wav2Vec2 (src/speech_encoder.py) et chercher dans l'index FAISS.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# Permet l'import de src/ quel que soit le répertoire d'exécution
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import faiss_index as fi  # noqa: E402  (import relatif utils/)

_EMBEDDINGS_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "embeddings", "text_embeddings.npy")


# ---------------------------------------------------------------------------
# Recherche S2R : audio brut → top-k documents (OBJECTIF DU PROJET)
# ---------------------------------------------------------------------------

def search_by_audio(
    audio_path: str,
    k: int = 5,
    embeddings_path: str = _EMBEDDINGS_DEFAULT,
    model_name: str = "facebook/wav2vec2-base-960h",
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Recherche les k documents les plus pertinents pour une requête audio.

    Pipeline S2R (sans transcription ASR) :
        fichier .wav → Wav2Vec2 → embedding 768D → FAISS → top-k

    Paramètres
    ----------
    audio_path : str
        Chemin vers le fichier audio .wav de la requête.
    k : int
        Nombre de résultats à retourner.

    Retourne
    --------
    (indices, scores) : tableaux numpy de forme (1, k)
    """
    from src.speech_encoder import speech_to_embedding

    query_embedding = speech_to_embedding(
        audio_path,
        model_name=model_name,
        device=device,
    ).astype("float32").reshape(1, -1)

    # Normaliser L2 la requête (cohérent avec IndexFlatIP)
    norm = np.linalg.norm(query_embedding)
    if norm > 1e-9:
        query_embedding = query_embedding / norm

    index = fi.getIndex(embeddings_path)
    scores, indices = index.search(query_embedding, k)
    return indices, scores


# ---------------------------------------------------------------------------
# Recherche texte-texte : prototype initial (NE PAS utiliser en production S2R)
# ---------------------------------------------------------------------------

def search_by_text(
    query: str,
    k: int = 5,
    embeddings_path: str = _EMBEDDINGS_DEFAULT,
    text_model: str = "all-MiniLM-L6-v2",
) -> tuple[np.ndarray, np.ndarray]:
    """Recherche texte-texte (prototype de validation, sans audio).

    ATTENTION : ce mode n'est PAS du Speech-to-Retrieval. Il existe pour
    valider l'index FAISS et le corpus. En production, utiliser search_by_audio().

    Retourne
    --------
    (indices, scores) : tableaux numpy de forme (1, k)
    """
    model = SentenceTransformer(text_model)
    query_embedding = model.encode([query]).astype("float32")  # (1, D)

    norm = np.linalg.norm(query_embedding)
    if norm > 1e-9:
        query_embedding = query_embedding / norm

    index = fi.getIndex(embeddings_path)
    scores, indices = index.search(query_embedding, k)
    return indices, scores


# ---------------------------------------------------------------------------
# Alias de compatibilité avec l'ancien code
# ---------------------------------------------------------------------------

def search(query: str, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Alias texte-texte conservé pour rétrocompatibilité."""
    return search_by_text(query, k=k)
