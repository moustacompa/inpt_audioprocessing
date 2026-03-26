"""
create_embeddings.py — Génération des embeddings du corpus (côté TEXTE et côté AUDIO).

Deux fonctions principales :
  1. create_text_embeddings()  → embeddings des documents texte du corpus
                                  (côté INDEXÉ dans FAISS, pour la recherche)
  2. create_audio_embeddings() → embeddings audio des fichiers .wav
                                  (côté REQUÊTE pour S2R, via Wav2Vec2 sans ASR)

NOTE SUR L'ARCHITECTURE S2R :
  - Les embeddings TEXTE sont indexés dans FAISS (corpus côté document)
  - Les embeddings AUDIO sont produits à la volée lors d'une requête (côté requête)
  - Le dual encoder (training/) apprend à aligner les deux espaces
  - Sans dual encoder entraîné, les deux espaces ont des dimensions différentes
    (384 pour MiniLM vs 768 pour Wav2Vec2), ce qui empêche la recherche directe.
  - Après entraînement, utiliser export_text_embeddings.py pour avoir des
    embeddings texte dans le même espace que les embeddings audio.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CORPUS_PATH = os.path.join(_ROOT, "data", "documents", "ss-corpus-fr.tsv")
AUDIO_FOLDER = os.path.join(_ROOT, "data", "audio_queries")
EMBEDDINGS_DIR = os.path.join(_ROOT, "embeddings")


# ---------------------------------------------------------------------------
# Côté TEXTE : embeddings des documents du corpus (indexés dans FAISS)
# ---------------------------------------------------------------------------

def create_text_embeddings(
    corpus_path: str = CORPUS_PATH,
    output_path: str = os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"),
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode les documents texte du corpus avec SentenceTransformer.

    Ces embeddings constituent le côté INDEXÉ du système de retrieval.
    Ils sont cherchés par les embeddings de requête (audio ou texte).

    Retourne
    --------
    np.ndarray de forme (N_docs, D)
    """
    tmp = pd.read_csv(corpus_path, sep="\t")
    # Utiliser le champ 'prompt' (questions) comme représentation du document
    # La colonne 'transcription' contient les réponses (peut être absente)
    data = tmp[tmp["prompt"].notna()].drop_duplicates(subset=["prompt"])
    sentences = data["prompt"].tolist()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Embeddings texte sauvegardés : {output_path} | shape={embeddings.shape}")
    return embeddings


def create_audio_embeddings(
    audio_folder: str = AUDIO_FOLDER,
    output_path: str = os.path.join(EMBEDDINGS_DIR, "audio_embeddings.npy"),
    manifest_path: str = os.path.join(EMBEDDINGS_DIR, "audio_embeddings_index.csv"),
    model_name: str = "facebook/wav2vec2-base-960h",
    device: str = "cpu",
) -> np.ndarray:
    """Encode les fichiers audio du corpus avec Wav2Vec2, SANS transcription.

    C'est le cœur du Speech-to-Retrieval : aucun ASR n'est impliqué.
    Les fichiers .wav sont directement convertis en vecteurs 768D.

    Ces embeddings sont utilisés comme requêtes pour chercher dans l'index
    FAISS (côté documents texte).

    Retourne
    --------
    np.ndarray de forme (N_audio, 768)
    """
    from src.speech_encoder import batch_speech_to_embeddings

    wav_files = sorted([
        f for f in os.listdir(audio_folder)
        if f.lower().endswith(".wav")
    ])

    if not wav_files:
        raise FileNotFoundError(
            f"Aucun fichier .wav trouvé dans : {audio_folder}\n"
            "Vérifiez que les fichiers audio ont été convertis en .wav (16 kHz, mono)."
        )

    audio_paths = [os.path.join(audio_folder, f) for f in wav_files]
    print(f"Encodage de {len(audio_paths)} fichiers audio via {model_name} (sans ASR)...")

    embeddings = batch_speech_to_embeddings(
        audio_paths,
        model_name=model_name,
        device=device,
        batch_size=8,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    # Sauvegarder le manifeste (audio_id → index de ligne)
    manifest = pd.DataFrame({
        "audio_id": [os.path.splitext(f)[0] for f in wav_files],
        "audio_path": audio_paths,
        "embedding_index": list(range(len(wav_files))),
    })
    manifest.to_csv(manifest_path, index=False)

    print(f"Embeddings audio sauvegardés : {output_path} | shape={embeddings.shape}")
    print(f"Manifeste sauvegardé         : {manifest_path}")
    return embeddings


# ---------------------------------------------------------------------------
# Utilitaires (compatibilité avec utils/test.py)
# ---------------------------------------------------------------------------

def getSentences(corpus_path: str = CORPUS_PATH) -> list[str]:
    """Retourne la liste des phrases/prompts du corpus (pour affichage des résultats)."""
    tmp = pd.read_csv(corpus_path, sep="\t")
    data = tmp[tmp["prompt"].notna()].drop_duplicates(subset=["prompt"])
    return data["prompt"].tolist()


def getTranscriptions(corpus_path: str = CORPUS_PATH) -> list[str]:
    """Retourne les transcriptions disponibles (peuvent être vides)."""
    tmp = pd.read_csv(corpus_path, sep="\t")
    data = tmp[tmp["transcription"].notna()]
    return data["transcription"].tolist()


# ---------------------------------------------------------------------------
# Exécution directe : génère les deux types d'embeddings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Génération des embeddings texte ===")
    create_text_embeddings()

    print("\n=== Génération des embeddings audio (S2R, sans ASR) ===")
    create_audio_embeddings()
