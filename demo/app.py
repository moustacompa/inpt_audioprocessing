"""
demo/app.py — P4: Interface Gradio Speech-to-Retrieval.

Lancer :
    python demo/app.py
    python demo/app.py --checkpoint models/dual_encoder_mpnet/best_model.pt

Interface :
    - Entrée  : microphone ou fichier .wav
    - Sortie  : tableau des top-5 documents pertinents avec scores

Pipeline S2R (sans transcription ASR) :
    audio .wav → Wav2Vec2 → [dual encoder optionnel] → FAISS → top-k docs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.speech_encoder import SpeechEncoder, _load_wav

# ---------------------------------------------------------------------------
# Configuration par défaut (peut être surchargée via arguments CLI)
# ---------------------------------------------------------------------------

DEFAULT_TEXT_EMBEDDINGS = str(_ROOT / "embeddings" / "text_chunk_embeddings.npy")
DEFAULT_MANIFEST = str(_ROOT / "embeddings" / "text_chunk_manifest.csv")
DEFAULT_CHECKPOINT = str(_ROOT / "models" / "dual_encoder_mpnet" / "best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Chargement des modèles (singleton)
# ---------------------------------------------------------------------------

_speech_encoder: SpeechEncoder | None = None
_dual_encoder = None
_faiss_index: faiss.Index | None = None
_manifest: pd.DataFrame | None = None


def load_models(checkpoint_path: str | None = None) -> None:
    global _speech_encoder, _dual_encoder, _faiss_index, _manifest

    print("Chargement SpeechEncoder (wav2vec2)...")
    _speech_encoder = SpeechEncoder(frozen=True).eval().to(DEVICE)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Chargement dual encoder : {checkpoint_path}")
        from training.models import DualEncoderModel
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        _dual_encoder = DualEncoderModel(**ckpt["config"])
        _dual_encoder.load_state_dict(ckpt["state_dict"])
        _dual_encoder.eval().to(DEVICE)
        text_emb_path = DEFAULT_TEXT_EMBEDDINGS
    else:
        print("Dual encoder non disponible → mode prototype (wav2vec2 brut).")
        _dual_encoder = None
        # En mode prototype, chercher dans les embeddings audio pré-calculés si disponibles
        text_emb_path = str(_ROOT / "embeddings" / "audio_embeddings.npy")
        if not Path(text_emb_path).exists():
            text_emb_path = DEFAULT_TEXT_EMBEDDINGS

    manifest_path = DEFAULT_MANIFEST
    if not Path(manifest_path).exists():
        manifest_path = str(_ROOT / "embeddings" / "audio_embeddings_index.csv")

    if Path(text_emb_path).exists() and Path(manifest_path).exists():
        print(f"Construction index FAISS : {text_emb_path}")
        embeddings = np.load(text_emb_path).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        _faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        _faiss_index.add(embeddings)
        _manifest = pd.read_csv(manifest_path)
        print(f"Index prêt : {_faiss_index.ntotal} documents")
    else:
        print(f"Embeddings introuvables ({text_emb_path}). Exécutez d'abord :")
        print("  python utils/create_embeddings.py")
        print("  ou entraînez le dual encoder puis exportez les embeddings texte.")


# ---------------------------------------------------------------------------
# Fonction de recherche principale (appelée par Gradio)
# ---------------------------------------------------------------------------

def search_from_audio(audio_path: str, k: int = 5) -> pd.DataFrame:
    """Pipeline S2R complet : fichier audio → DataFrame de résultats."""
    if audio_path is None:
        return pd.DataFrame({"Erreur": ["Veuillez fournir un fichier audio."]})

    if _speech_encoder is None:
        return pd.DataFrame({"Erreur": ["Modèles non chargés. Relancez l'application."]})

    # 1. Encoder l'audio sans ASR
    waveform = _load_wav(audio_path).to(DEVICE)
    with torch.no_grad():
        audio_emb = _speech_encoder(waveform.unsqueeze(0))  # [1, 768]

        if _dual_encoder is not None:
            audio_emb = _dual_encoder.encode_audio(audio_emb)  # [1, proj_dim]

    query = audio_emb.squeeze(0).cpu().numpy().astype("float32")
    norm = np.linalg.norm(query)
    if norm > 1e-9:
        query = query / norm

    # 2. Vérifier la cohérence de dimension
    if _faiss_index is None:
        return pd.DataFrame({"Erreur": ["Index FAISS non disponible."]})

    if query.shape[0] != _faiss_index.d:
        return pd.DataFrame({
            "Erreur": [
                f"Dimension incompatible : requête={query.shape[0]}, index={_faiss_index.d}. "
                "Entraînez le dual encoder (voir training/)."
            ]
        })

    # 3. Recherche FAISS
    scores, indices = _faiss_index.search(query.reshape(1, -1), min(k, _faiss_index.ntotal))

    # 4. Formater les résultats
    rows = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        row = _manifest.iloc[int(idx)]
        text = row.get("text", row.get("prompt", row.get("audio_path", str(dict(row)))))
        rows.append({
            "Rang": rank,
            "Score": round(float(score), 4),
            "Document": str(text)[:300],
            "chunk_id": str(row.get("chunk_id", row.get("audio_id", ""))),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Speech-to-Retrieval Demo") as demo:
        gr.Markdown(
            """
            # Speech-to-Retrieval (S2R) — Demo
            **Recherche documentaire directement depuis la voix, sans transcription ASR.**

            Pipeline : `audio .wav` → `Wav2Vec2` → `embedding` → `FAISS` → `top-k documents`

            *Projet INPT — Deep Learning — Français/Anglais*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Requête vocale",
                    type="filepath",
                    sources=["microphone", "upload"],
                )
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Nombre de résultats (k)",
                )
                search_btn = gr.Button("Rechercher", variant="primary")

            with gr.Column(scale=2):
                results_table = gr.Dataframe(
                    label="Documents pertinents",
                    headers=["Rang", "Score", "Document", "chunk_id"],
                    wrap=True,
                )

        search_btn.click(
            fn=search_from_audio,
            inputs=[audio_input, k_slider],
            outputs=results_table,
        )

        gr.Markdown(
            """
            ---
            **Notes :**
            - En mode *prototype* (sans dual encoder entraîné) : les résultats reflètent
              la similarité brute dans l'espace Wav2Vec2, non aligné avec l'espace texte.
            - Après entraînement du dual encoder (P3), relancez avec `--checkpoint`.
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Démo Gradio S2R")
    parser.add_argument("--checkpoint", default=None, help="Chemin vers best_model.pt (P3)")
    parser.add_argument("--share", action="store_true", help="Partager via lien public Gradio")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_models(checkpoint_path=args.checkpoint or DEFAULT_CHECKPOINT)
    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)
