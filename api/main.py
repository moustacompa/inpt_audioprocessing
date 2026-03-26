"""
api/main.py — P4: API FastAPI Speech-to-Retrieval.

Endpoints :
    POST /search        : audio base64 → top-k documents
    POST /search/file   : fichier .wav upload → top-k documents
    GET  /health        : vérification que l'API est opérationnelle

Lancer :
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Variables d'environnement (optionnelles) :
    S2R_CHECKPOINT      : chemin vers best_model.pt (dual encoder P3)
    S2R_TEXT_EMBEDDINGS : chemin vers text_chunk_embeddings.npy
    S2R_MANIFEST        : chemin vers text_chunk_manifest.csv
    S2R_DEVICE          : 'cpu' ou 'cuda'
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration via variables d'environnement
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get("S2R_CHECKPOINT", "models/dual_encoder_mpnet/best_model.pt")
TEXT_EMBEDDINGS_PATH = os.environ.get(
    "S2R_TEXT_EMBEDDINGS", "embeddings/text_chunk_embeddings.npy"
)
MANIFEST_PATH = os.environ.get("S2R_MANIFEST", "embeddings/text_chunk_manifest.csv")
DEVICE = os.environ.get("S2R_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# État global (chargé au démarrage)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _build_index(embeddings: np.ndarray) -> faiss.Index:
    embs = embeddings.astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, a_min=1e-9, a_max=None)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def _load_dual_encoder(checkpoint_path: str) -> Any:
    from training.models import DualEncoderModel

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = DualEncoderModel(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles et l'index au démarrage de l'API."""
    from src.speech_encoder import SpeechEncoder

    print(f"Chargement SpeechEncoder (wav2vec2)...")
    _state["speech_encoder"] = SpeechEncoder(frozen=True).eval().to(DEVICE)

    if Path(CHECKPOINT).exists():
        print(f"Chargement dual encoder : {CHECKPOINT}")
        _state["dual_encoder"] = _load_dual_encoder(CHECKPOINT)
    else:
        print(f"Checkpoint introuvable ({CHECKPOINT}). Mode prototype (wav2vec2 brut).")
        _state["dual_encoder"] = None

    if Path(TEXT_EMBEDDINGS_PATH).exists() and Path(MANIFEST_PATH).exists():
        print(f"Chargement index FAISS : {TEXT_EMBEDDINGS_PATH}")
        embeddings = np.load(TEXT_EMBEDDINGS_PATH)
        _state["index"] = _build_index(embeddings)
        _state["manifest"] = pd.read_csv(MANIFEST_PATH)
        _state["embedding_dim"] = embeddings.shape[1]
        print(f"Index prêt : {_state['index'].ntotal} documents | dim={_state['embedding_dim']}")
    else:
        print("Embeddings texte introuvables. Indexation désactivée.")
        _state["index"] = None
        _state["manifest"] = None
        _state["embedding_dim"] = None

    yield
    _state.clear()


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Speech-to-Retrieval API",
    description=(
        "Recherche documentaire directement depuis la voix, sans transcription ASR. "
        "Pipeline : audio .wav → Wav2Vec2 → embedding → FAISS → top-k documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas de données
# ---------------------------------------------------------------------------

class AudioBase64Request(BaseModel):
    audio_base64: str
    k: int = 5


class DocumentResult(BaseModel):
    rank: int
    score: float
    chunk_id: str | None = None
    document_id: str | None = None
    text: str | None = None


class SearchResponse(BaseModel):
    results: list[DocumentResult]
    query_embedding_dim: int
    index_size: int


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _wav_bytes_to_embedding(wav_bytes: bytes) -> np.ndarray:
    """Convertit des bytes audio en embedding, via SpeechEncoder + optionnellement DualEncoder."""
    import torchaudio
    from src.speech_encoder import _load_wav

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        from src.speech_encoder import _load_wav
        waveform = _load_wav(tmp_path).to(DEVICE)
        speech_enc: torch.nn.Module = _state["speech_encoder"]

        with torch.no_grad():
            audio_emb = speech_enc(waveform.unsqueeze(0))  # [1, 768]

        dual_enc = _state.get("dual_encoder")
        if dual_enc is not None:
            with torch.no_grad():
                audio_emb = dual_enc.encode_audio(audio_emb)  # [1, proj_dim]

        return audio_emb.squeeze(0).cpu().numpy()
    finally:
        os.unlink(tmp_path)


def _do_search(query_emb: np.ndarray, k: int) -> list[DocumentResult]:
    index: faiss.Index = _state.get("index")
    manifest: pd.DataFrame = _state.get("manifest")

    if index is None:
        raise HTTPException(status_code=503, detail="Index FAISS non disponible.")

    emb_dim = _state.get("embedding_dim")
    if query_emb.shape[0] != emb_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dimension de l'embedding ({query_emb.shape[0]}) "
                f"incompatible avec l'index ({emb_dim}). "
                "Entraînez le dual encoder ou recréez l'index."
            ),
        )

    query = query_emb.astype("float32").reshape(1, -1)
    norm = np.linalg.norm(query)
    if norm > 1e-9:
        query /= norm

    scores, indices = index.search(query, min(k, index.ntotal))
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        row = manifest.iloc[int(idx)]
        results.append(DocumentResult(
            rank=rank,
            score=float(score),
            chunk_id=str(row.get("chunk_id", "")),
            document_id=str(row.get("document_id", "")),
            text=str(row.get("text", row.get("prompt", ""))),
        ))
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    """Vérifie que l'API est opérationnelle."""
    return {
        "status": "ok",
        "dual_encoder_loaded": _state.get("dual_encoder") is not None,
        "index_size": _state["index"].ntotal if _state.get("index") else 0,
        "device": DEVICE,
    }


@app.post("/search", response_model=SearchResponse)
def search_base64(request: AudioBase64Request) -> SearchResponse:
    """Recherche à partir d'un audio encodé en base64.

    Envoyer un fichier .wav encodé en base64 dans le champ `audio_base64`.
    """
    try:
        wav_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio_base64 invalide.")

    query_emb = _wav_bytes_to_embedding(wav_bytes)
    results = _do_search(query_emb, request.k)

    return SearchResponse(
        results=results,
        query_embedding_dim=int(query_emb.shape[0]),
        index_size=_state["index"].ntotal if _state.get("index") else 0,
    )


@app.post("/search/file", response_model=SearchResponse)
async def search_file(
    file: UploadFile = File(..., description="Fichier .wav de la requête vocale"),
    k: int = 5,
) -> SearchResponse:
    """Recherche à partir d'un fichier .wav uploadé directement."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .wav sont acceptés.")

    wav_bytes = await file.read()
    query_emb = _wav_bytes_to_embedding(wav_bytes)
    results = _do_search(query_emb, k)

    return SearchResponse(
        results=results,
        query_embedding_dim=int(query_emb.shape[0]),
        index_size=_state["index"].ntotal if _state.get("index") else 0,
    )
