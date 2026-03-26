"""
test.py — Tests du pipeline Speech-to-Retrieval.

Deux modes de test :
  1. test_audio_search()  → S2R réel : fichier .wav → top-k documents (OBJECTIF)
  2. test_text_search()   → prototype texte : requête texte → top-k documents

Lancer depuis le dossier utils/ :
    python test.py --mode audio --audio data/audio_queries/spontaneous-speech-fr-20300.wav
    python test.py --mode text  --query "Dix-huit heures, tout rond."
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import create_embeddings as ce
import search as s


def test_audio_search(audio_path: str, k: int = 5) -> None:
    """Test S2R réel : audio brut → top-k documents, sans ASR."""
    print(f"\n=== TEST S2R (audio → documents, sans transcription) ===")
    print(f"Fichier audio : {audio_path}")

    corpus = ce.getSentences()
    indices, scores = s.search_by_audio(audio_path, k=k)

    indices = indices[0]
    scores = scores[0]

    print(f"\nTop-{k} résultats :")
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        doc = corpus[idx] if idx < len(corpus) else "[index hors corpus]"
        print(f"  {rank}. [score={score:.4f}] idx={idx} → {doc[:120]}")


def test_text_search(query: str, k: int = 5) -> None:
    """Test prototype texte-texte (NE PAS confondre avec S2R)."""
    print(f"\n=== TEST PROTOTYPE TEXTE (non S2R) ===")
    print(f"Requête : {query!r}")

    corpus = ce.getSentences()
    indices, scores = s.search_by_text(query, k=k)

    indices = indices[0]
    scores = scores[0]

    print(f"\nTop-{k} résultats :")
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        doc = corpus[idx] if idx < len(corpus) else "[index hors corpus]"
        print(f"  {rank}. [score={score:.4f}] idx={idx} → {doc[:120]}")


def _find_first_wav() -> str | None:
    audio_dir = os.path.join(_ROOT, "data", "audio_queries")
    if not os.path.isdir(audio_dir):
        return None
    for f in sorted(os.listdir(audio_dir)):
        if f.lower().endswith(".wav"):
            return os.path.join(audio_dir, f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test du pipeline S2R")
    parser.add_argument(
        "--mode",
        choices=("audio", "text"),
        default="audio",
        help="Mode de recherche : 'audio' (S2R réel) ou 'text' (prototype)",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Chemin vers un fichier .wav pour le mode audio",
    )
    parser.add_argument(
        "--query",
        default="Dix-huit heures, tout rond.",
        help="Requête texte pour le mode text",
    )
    parser.add_argument("--k", type=int, default=5, help="Nombre de résultats")
    args = parser.parse_args()

    if args.mode == "audio":
        wav_path = args.audio or _find_first_wav()
        if wav_path is None:
            print("Aucun fichier .wav trouvé. Spécifiez --audio <chemin>.")
            sys.exit(1)
        test_audio_search(wav_path, k=args.k)
    else:
        test_text_search(args.query, k=args.k)
