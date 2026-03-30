"""
api/main.py — API FastAPI Speech-to-Retrieval (S2R)

Endpoints :
    GET  /health           etat de l'API
    POST /search/audio     fichier .wav (upload) → top-k documents
    POST /search/base64    audio base64 → top-k documents
    POST /search/text      requete texte → top-k documents (comparaison)
    GET  /examples         liste des fichiers audio de demo disponibles

Lancer :
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Variables d'environnement :
    S2R_CHECKPOINT        chemin vers best_model.pt
    S2R_TEXT_EMBEDDINGS   chemin vers text_chunk_embeddings.npy
    S2R_MANIFEST          chemin vers text_chunk_manifest.csv
    S2R_DEVICE            'cpu' ou 'cuda'
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT        = os.environ.get("S2R_CHECKPOINT",      str(_ROOT / "models/dual_encoder_mpnet/best_model.pt"))
TEXT_EMBEDDINGS   = os.environ.get("S2R_TEXT_EMBEDDINGS", str(_ROOT / "embeddings/text_chunk_embeddings.npy"))
MANIFEST_PATH     = os.environ.get("S2R_MANIFEST",        str(_ROOT / "embeddings/text_chunk_manifest.csv"))
DEVICE            = os.environ.get("S2R_DEVICE",          "cuda" if torch.cuda.is_available() else "cpu")
AUDIO_QUERIES_DIR = str(_ROOT / "data" / "audio_queries")

# ---------------------------------------------------------------------------
# Etat global
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


def _load_checkpoint(path: str):
    from training.models import DualEncoderModel
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DualEncoderModel(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model, ckpt.get("best_metrics", {})


def _build_faiss_index(npy_path: str):
    embs = np.load(npy_path).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= np.clip(norms, 1e-9, None)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, embs.shape[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.speech_encoder import SpeechEncoder
    print(f"[S2R] Chargement SpeechEncoder (Wav2Vec2)...")
    _state["speech_encoder"] = SpeechEncoder(frozen=True).eval().to(DEVICE)

    if Path(CHECKPOINT).exists():
        print(f"[S2R] Chargement dual encoder : {CHECKPOINT}")
        model, metrics = _load_checkpoint(CHECKPOINT)
        _state["dual_encoder"] = model
        _state["train_metrics"] = metrics
        print(f"[S2R] Metriques d'entrainement : {metrics}")
    else:
        print(f"[S2R] Checkpoint introuvable. Mode prototype (espace Wav2Vec2 brut).")
        _state["dual_encoder"] = None
        _state["train_metrics"] = {}

    if Path(TEXT_EMBEDDINGS).exists() and Path(MANIFEST_PATH).exists():
        print(f"[S2R] Construction index FAISS : {TEXT_EMBEDDINGS}")
        index, dim = _build_faiss_index(TEXT_EMBEDDINGS)
        _state["index"] = index
        _state["index_dim"] = dim
        _state["manifest"] = pd.read_csv(MANIFEST_PATH)
        print(f"[S2R] Index pret : {index.ntotal} vecteurs | dim={dim}")
    else:
        print(f"[S2R] Embeddings texte introuvables ({TEXT_EMBEDDINGS}).")
        _state["index"] = None

    yield
    _state.clear()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Speech-to-Retrieval API",
    description=(
        "Recherche documentaire directement depuis la voix, sans transcription ASR.\n\n"
        "Pipeline : `.wav` → Wav2Vec2 → [dual encoder] → FAISS → top-k documents"
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AudioBase64Request(BaseModel):
    audio_base64: str
    k: int = 5


class TextRequest(BaseModel):
    query: str
    k: int = 5


class DocumentResult(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    text: str


class SearchResponse(BaseModel):
    results: list[DocumentResult]
    mode: str
    query_dim: int
    index_size: int


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------
def _encode_audio_bytes(wav_bytes: bytes) -> np.ndarray:
    """bytes WAV → embedding numpy via SpeechEncoder + optionnel DualEncoder."""
    from src.speech_encoder import _load_wav

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        waveform = _load_wav(tmp).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = _state["speech_encoder"](waveform)           # [1, 768]
            dual = _state.get("dual_encoder")
            if dual is not None:
                emb = dual.encode_audio(emb)                   # [1, proj_dim]
        return emb.squeeze(0).cpu().numpy().astype("float32")
    finally:
        os.unlink(tmp)


def _encode_text_query(query: str) -> np.ndarray:
    """Texte → embedding via SentenceTransformer + optionnel text_projection."""
    from sentence_transformers import SentenceTransformer

    dual = _state.get("dual_encoder")
    if dual is not None and dual.text_encoder is not None:
        # Utilise le text encoder du dual encoder
        from transformers import AutoTokenizer
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        tok_dir = Path(CHECKPOINT).parent / "tokenizer"
        tok_src = tok_dir if tok_dir.exists() else ckpt["config"]["text_model_name"]
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            emb = dual.encode_text(
                enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
            )
        return emb.squeeze(0).cpu().numpy().astype("float32")
    else:
        # Fallback : SentenceTransformer sans projection
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(query, normalize_embeddings=True).astype("float32")
        return emb


def _do_search(query_emb: np.ndarray, k: int) -> tuple[list[DocumentResult], str]:
    index: faiss.Index = _state.get("index")
    if index is None:
        raise HTTPException(503, "Index FAISS non disponible. Lancez l'etape 3 du pipeline.")

    if query_emb.shape[0] != _state["index_dim"]:
        raise HTTPException(
            422,
            f"Dimension incompatible : requete={query_emb.shape[0]}, "
            f"index={_state['index_dim']}. "
            "Exportez les embeddings texte apres l'entrainement (etape 3)."
        )

    q = query_emb.reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm

    scores, indices = index.search(q, min(k, index.ntotal))
    manifest: pd.DataFrame = _state["manifest"]
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        row = manifest.iloc[int(idx)]
        results.append(DocumentResult(
            rank=rank,
            score=round(float(score), 4),
            chunk_id=str(row.get("chunk_id", "")),
            document_id=str(row.get("document_id", "")),
            text=str(row.get("text", row.get("prompt", "")))[:500],
        ))

    mode = "dual_encoder" if _state.get("dual_encoder") else "prototype_wav2vec2"
    return results, mode


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Etat de l'API")
def health():
    dual = _state.get("dual_encoder")
    index = _state.get("index")
    return {
        "status": "ok",
        "device": DEVICE,
        "dual_encoder_loaded": dual is not None,
        "fast_mode": dual is not None and dual.text_encoder is None,
        "index_size": index.ntotal if index else 0,
        "index_dim": _state.get("index_dim"),
        "train_metrics": _state.get("train_metrics", {}),
    }


@app.post("/search/audio", response_model=SearchResponse, summary="Recherche par fichier audio")
async def search_audio(
    file: UploadFile = File(..., description="Fichier .wav (16 kHz mono recommande)"),
    k: int = Query(5, ge=1, le=50, description="Nombre de resultats"),
):
    """Upload un fichier `.wav` et retourne les k documents les plus proches."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(400, "Format audio non supporte. Utilisez .wav de preference.")

    wav_bytes = await file.read()
    query_emb = _encode_audio_bytes(wav_bytes)
    results, mode = _do_search(query_emb, k)
    return SearchResponse(results=results, mode=mode,
                          query_dim=int(query_emb.shape[0]),
                          index_size=_state["index"].ntotal)


@app.post("/search/base64", response_model=SearchResponse, summary="Recherche par audio base64")
def search_base64(request: AudioBase64Request):
    """Envoie un fichier `.wav` encode en base64."""
    try:
        wav_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(400, "audio_base64 invalide (base64 mal forme).")

    query_emb = _encode_audio_bytes(wav_bytes)
    results, mode = _do_search(query_emb, request.k)
    return SearchResponse(results=results, mode=mode,
                          query_dim=int(query_emb.shape[0]),
                          index_size=_state["index"].ntotal)


@app.post("/search/text", response_model=SearchResponse, summary="Recherche par texte (comparaison)")
def search_text(request: TextRequest):
    """Recherche par texte ecrit. Utile pour comparer avec la recherche vocale."""
    query_emb = _encode_text_query(request.query)
    results, mode = _do_search(query_emb, request.k)
    return SearchResponse(results=results, mode=f"text_{mode}",
                          query_dim=int(query_emb.shape[0]),
                          index_size=_state["index"].ntotal)


@app.get("/examples", summary="Fichiers audio de demonstration disponibles")
def list_examples():
    """Retourne la liste des fichiers .wav de demo dans data/audio_queries/."""
    folder = Path(AUDIO_QUERIES_DIR)
    if not folder.exists():
        return {"files": []}
    files = sorted(f.name for f in folder.glob("*.wav"))
    return {"files": files, "folder": AUDIO_QUERIES_DIR}
