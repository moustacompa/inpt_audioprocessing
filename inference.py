"""
inference.py — Pipeline complet Speech-to-Retrieval (S2R).

Usage (après entraînement du dual encoder) :
    python inference.py \\
        --audio data/audio_queries/spontaneous-speech-fr-20300.wav \\
        --checkpoint models/dual_encoder_mpnet/best_model.pt \\
        --text-embeddings embeddings/text_chunk_embeddings.npy \\
        --manifest embeddings/text_chunk_manifest.csv

Usage sans checkpoint (prototype wav2vec2 brut) :
    python inference.py \\
        --audio data/audio_queries/spontaneous-speech-fr-20300.wav \\
        --text-embeddings embeddings/audio_embeddings.npy \\
        --manifest embeddings/audio_embeddings_index.csv \\
        --no-dual-encoder
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from src.speech_encoder import speech_to_embedding


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Construit un index FAISS (Inner Product = cosine sur vecteurs normalisés)."""
    embs = embeddings.astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, a_min=1e-9, a_max=None)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def encode_audio_with_dual_encoder(
    audio_path: str,
    checkpoint_path: str,
    device: str,
) -> np.ndarray:
    """Encode un fichier audio via le dual encoder entraîné (P3).

    Pipeline :
        wav → wav2vec2 → audio_embedding → audio_projection → vecteur aligné
    """
    from training.models import DualEncoderModel
    from src.speech_encoder import SpeechEncoder, _load_wav

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    # 1. Encoder wav2vec2 → embedding brut
    speech_enc = SpeechEncoder(frozen=True).to(device)
    speech_enc.eval()
    waveform = _load_wav(audio_path).to(device)
    with torch.no_grad():
        raw_audio_emb = speech_enc(waveform.unsqueeze(0))  # [1, 768]

    # 2. Projection du dual encoder → espace partagé
    dual_model = DualEncoderModel(**config)
    dual_model.load_state_dict(checkpoint["state_dict"])
    dual_model.eval().to(device)

    with torch.no_grad():
        projected = dual_model.encode_audio(raw_audio_emb)  # [1, proj_dim]

    return projected.squeeze(0).cpu().numpy()  # (proj_dim,)


def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    manifest: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Cherche les k documents les plus proches dans l'index FAISS.

    Retourne un DataFrame avec les colonnes du manifeste + score.
    """
    query = query_embedding.astype("float32").reshape(1, -1)
    norm = np.linalg.norm(query)
    if norm > 1e-9:
        query = query / norm

    scores, indices = index.search(query, k)
    results = manifest.iloc[indices[0]].copy()
    results["score"] = scores[0]
    results = results.reset_index(drop=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech-to-Retrieval : audio → top-k documents")
    parser.add_argument("--audio", required=True, help="Fichier .wav de la requête vocale")
    parser.add_argument(
        "--text-embeddings",
        default="embeddings/text_chunk_embeddings.npy",
        help="Embeddings texte du corpus (.npy)",
    )
    parser.add_argument(
        "--manifest",
        default="embeddings/text_chunk_manifest.csv",
        help="Manifeste des chunks texte (.csv)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint du dual encoder entraîné (best_model.pt). "
             "Si absent, utilise wav2vec2 brut (prototype).",
    )
    parser.add_argument("--k", type=int, default=5, help="Nombre de résultats à retourner")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Fichier audio : {args.audio}")
    print(f"Embeddings    : {args.text_embeddings}")
    print(f"Manifeste     : {args.manifest}")
    print(f"Checkpoint    : {args.checkpoint or 'non fourni (prototype brut)'}")
    print()

    # 1. Encoder la requête audio
    if args.checkpoint and Path(args.checkpoint).exists():
        print("Encodage via dual encoder entraîné (P3)...")
        query_emb = encode_audio_with_dual_encoder(
            args.audio,
            args.checkpoint,
            args.device,
        )
    else:
        print("Encodage via Wav2Vec2 brut (prototype, sans dual encoder)...")
        query_emb = speech_to_embedding(args.audio, device=args.device)

    print(f"Embedding audio shape : {query_emb.shape}")

    # 2. Charger l'index texte
    text_embeddings = np.load(args.text_embeddings)
    manifest = pd.read_csv(args.manifest)
    index = build_faiss_index(text_embeddings)
    print(f"Index FAISS : {index.ntotal} documents | dim={text_embeddings.shape[1]}")

    # 3. Vérifier la cohérence de dimension
    if query_emb.shape[0] != text_embeddings.shape[1]:
        print(
            f"\nATTENTION : dimension de la requête ({query_emb.shape[0]}) "
            f"≠ dimension des embeddings texte ({text_embeddings.shape[1]}).\n"
            "Les espaces ne sont pas alignés. "
            "Entraînez le dual encoder (training/) et utilisez --checkpoint."
        )
        return

    # 4. Recherche top-k
    results = search(query_emb, index, manifest, k=args.k)

    print(f"\nTop-{args.k} documents pertinents :\n")
    for rank, row in results.iterrows():
        score = row.get("score", float("nan"))
        text = row.get("text", row.get("prompt", str(dict(row))))
        print(f"  {rank + 1}. [score={score:.4f}]")
        print(f"     {str(text)[:200]}")
        print()


if __name__ == "__main__":
    main()
