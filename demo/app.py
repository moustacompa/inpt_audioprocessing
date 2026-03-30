"""
demo/app.py — Interface Gradio Speech-to-Retrieval

Lancer :
    python demo/app.py
    python demo/app.py --checkpoint models/dual_encoder_mpnet/best_model.pt
    python demo/app.py --checkpoint models/dual_encoder_mpnet/best_model.pt --share
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch

from src.speech_encoder import SpeechEncoder, _load_wav

# ---------------------------------------------------------------------------
# Chemins par defaut
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT      = str(_ROOT / "models" / "dual_encoder_mpnet" / "best_model.pt")
DEFAULT_TEXT_EMBEDDINGS = str(_ROOT / "embeddings" / "text_chunk_embeddings.npy")
DEFAULT_MANIFEST        = str(_ROOT / "embeddings" / "text_chunk_manifest.csv")
DEFAULT_QUERIES_DIR     = str(_ROOT / "data" / "audio_queries")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Etat global (charge une seule fois)
# ---------------------------------------------------------------------------
_speech_encoder: SpeechEncoder | None = None
_dual_encoder = None
_index: faiss.Index | None = None
_manifest: pd.DataFrame | None = None
_status_info: dict = {}


def _load_checkpoint(path: str):
    from training.models import DualEncoderModel
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DualEncoderModel(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model, ckpt.get("best_metrics", {})


def load_models(checkpoint_path: str | None = None) -> str:
    global _speech_encoder, _dual_encoder, _index, _manifest, _status_info

    lines = []

    # 1. Speech encoder (Wav2Vec2)
    lines.append("Chargement Wav2Vec2...")
    _speech_encoder = SpeechEncoder(frozen=True).eval().to(DEVICE)
    lines.append(f"  Wav2Vec2 OK (sortie 768D, device={DEVICE})")

    # 2. Dual encoder (optionnel)
    ckpt_path = checkpoint_path or DEFAULT_CHECKPOINT
    if Path(ckpt_path).exists():
        lines.append(f"Chargement dual encoder : {Path(ckpt_path).name}")
        _dual_encoder, metrics = _load_checkpoint(ckpt_path)
        fast = _dual_encoder.text_encoder is None
        proj_dim = _dual_encoder.audio_projection[1].out_features
        lines.append(f"  Dual encoder OK | mode={'rapide' if fast else 'complet'} | projection={proj_dim}D")
        if metrics:
            lines.append(f"  Metriques : Recall@5={metrics.get('Recall@5', 0):.3f} | "
                         f"Recall@10={metrics.get('Recall@10', 0):.3f} | "
                         f"MRR={metrics.get('MRR', 0):.3f}")
        _status_info["mode"] = "Dual Encoder entraine"
        _status_info["proj_dim"] = proj_dim
    else:
        lines.append(f"Checkpoint introuvable -> mode prototype (Wav2Vec2 brut)")
        _dual_encoder = None
        _status_info["mode"] = "Prototype (Wav2Vec2 brut, espaces non alignes)"
        _status_info["proj_dim"] = 768

    # 3. Index FAISS
    emb_path = DEFAULT_TEXT_EMBEDDINGS
    manifest_path = DEFAULT_MANIFEST
    if Path(emb_path).exists() and Path(manifest_path).exists():
        lines.append(f"Construction index FAISS...")
        embs = np.load(emb_path).astype("float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs /= np.clip(norms, 1e-9, None)
        _index = faiss.IndexFlatIP(embs.shape[1])
        _index.add(embs)
        _manifest = pd.read_csv(manifest_path)
        lines.append(f"  Index OK : {_index.ntotal} documents | dim={embs.shape[1]}D")
        _status_info["index_size"] = _index.ntotal
        _status_info["index_dim"] = embs.shape[1]
    else:
        lines.append(f"Embeddings texte introuvables : {emb_path}")
        lines.append("  -> Lancez l'etape 3 du pipeline (export_text_embeddings)")
        _index = None
        _status_info["index_size"] = 0

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logique de recherche
# ---------------------------------------------------------------------------
def _search(query_emb: np.ndarray, k: int) -> pd.DataFrame:
    if _index is None:
        return pd.DataFrame({"Erreur": [
            "Index FAISS non disponible. "
            "Lancez : python -m training.export_text_embeddings ..."
        ]})

    if query_emb.shape[0] != _index.d:
        return pd.DataFrame({"Erreur": [
            f"Dimension incompatible : requete={query_emb.shape[0]}D, index={_index.d}D.\n"
            "Le dual encoder doit etre entraine et les embeddings texte re-exportes."
        ]})

    q = query_emb.astype("float32").reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm

    scores, indices = _index.search(q, min(k, _index.ntotal))
    rows = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        row = _manifest.iloc[int(idx)]
        rows.append({
            "Rang": rank,
            "Score": round(float(score), 4),
            "Texte": str(row.get("text", row.get("prompt", "")))[:400],
            "chunk_id": str(row.get("chunk_id", "")),
            "document_id": str(row.get("document_id", "")),
        })
    return pd.DataFrame(rows)


def search_audio(audio_path: str | None, k: int) -> tuple[pd.DataFrame, str]:
    if audio_path is None:
        return pd.DataFrame(), "Aucun audio fourni."
    if _speech_encoder is None:
        return pd.DataFrame(), "Modeles non charges."

    waveform = _load_wav(audio_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _speech_encoder(waveform)
        if _dual_encoder is not None:
            emb = _dual_encoder.encode_audio(emb)

    query_emb = emb.squeeze(0).cpu().numpy()
    df = _search(query_emb, k)
    info = f"Embedding dim={query_emb.shape[0]} | mode={_status_info.get('mode', '?')}"
    return df, info


def search_text(query: str, k: int) -> tuple[pd.DataFrame, str]:
    if not query.strip():
        return pd.DataFrame(), "Requete vide."
    if _index is None:
        return pd.DataFrame(), "Index non disponible."

    from sentence_transformers import SentenceTransformer
    smodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = smodel.encode(query, normalize_embeddings=True).astype("float32")
    df = _search(emb, k)
    info = f"Mode texte (MiniLM, 384D)"
    return df, info


def load_example(filename: str) -> str | None:
    if not filename:
        return None
    path = Path(DEFAULT_QUERIES_DIR) / filename
    return str(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    mode_label = _status_info.get("mode", "Non charge")
    index_size = _status_info.get("index_size", 0)

    with gr.Blocks(
        title="Speech-to-Retrieval — Demo",
        theme=gr.themes.Soft(),
    ) as demo:

        # En-tete
        gr.Markdown("""
# Speech-to-Retrieval (S2R)
**Recherche documentaire directement depuis la voix — sans transcription ASR**

Pipeline : `audio .wav` → `Wav2Vec2` → `[projection]` → `FAISS` → `top-k documents`
        """)

        # Panneau de statut
        with gr.Row():
            gr.Markdown(f"""
| Composant | Statut |
|-----------|--------|
| Mode | **{mode_label}** |
| Index FAISS | **{index_size} documents** |
| Device | **{DEVICE}** |
            """)

        gr.Markdown("---")

        # Onglets
        with gr.Tabs():

            # --- Onglet 1 : Recherche vocale ---
            with gr.TabItem("Recherche vocale"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_in = gr.Audio(
                            label="Requete vocale",
                            type="filepath",
                            sources=["microphone", "upload"],
                        )

                        k_audio = gr.Slider(1, 20, value=5, step=1,
                                            label="Nombre de resultats (k)")

                        btn_audio = gr.Button("Rechercher", variant="primary", size="lg")

                        # Exemples rapides
                        example_files = sorted(
                            f.name for f in Path(DEFAULT_QUERIES_DIR).glob("*.wav")
                        ) if Path(DEFAULT_QUERIES_DIR).exists() else []

                        if example_files:
                            gr.Markdown("**Exemples audio :**")
                            example_dd = gr.Dropdown(
                                choices=example_files,
                                label="Charger un exemple",
                                value=None,
                            )
                            example_dd.change(
                                fn=load_example,
                                inputs=example_dd,
                                outputs=audio_in,
                            )

                    with gr.Column(scale=2):
                        audio_info = gr.Textbox(label="Info", lines=1, interactive=False)
                        audio_results = gr.Dataframe(
                            label="Documents retrouves",
                            headers=["Rang", "Score", "Texte", "chunk_id", "document_id"],
                            wrap=True,
                            row_count=5,
                        )

                btn_audio.click(
                    fn=search_audio,
                    inputs=[audio_in, k_audio],
                    outputs=[audio_results, audio_info],
                )

            # --- Onglet 2 : Recherche texte (comparaison) ---
            with gr.TabItem("Recherche texte (comparaison)"):
                gr.Markdown(
                    "_Recherche par texte ecrit — utile pour valider que l'index "
                    "contient les bons documents avant de tester la voix._"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        text_in = gr.Textbox(
                            label="Requete textuelle",
                            placeholder="Ex: customer support audio retrieval...",
                            lines=3,
                        )
                        k_text = gr.Slider(1, 20, value=5, step=1,
                                           label="Nombre de resultats (k)")
                        btn_text = gr.Button("Rechercher", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        text_info = gr.Textbox(label="Info", lines=1, interactive=False)
                        text_results = gr.Dataframe(
                            label="Documents retrouves",
                            headers=["Rang", "Score", "Texte", "chunk_id", "document_id"],
                            wrap=True,
                            row_count=5,
                        )

                btn_text.click(
                    fn=search_text,
                    inputs=[text_in, k_text],
                    outputs=[text_results, text_info],
                )

        # Notes
        gr.Markdown("""
---
**Notes :**
- **Mode prototype** : espaces audio et texte non alignes → resultats non pertinents.
- **Apres entrainement** : relancer avec `--checkpoint models/dual_encoder_mpnet/best_model.pt`.
- L'onglet *Recherche texte* utilise MiniLM (384D) — dimension independante du dual encoder.
        """)

    return demo


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo Gradio S2R")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log = load_models(checkpoint_path=args.checkpoint)
    print(log)
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)
