"""
export_text_embeddings.py — Exporte les embeddings texte dans l'espace partagé du dual encoder.

Supporte deux modes :
  - Mode complet  : le modele encode le texte directement (text_model_name non nul)
  - Mode rapide   : applique la text_projection sur des embeddings pre-calcules
                    (utiliser --precomputed-text-npy + --precomputed-text-manifest)

Usage mode complet :
    python -m training.export_text_embeddings \
        --text-chunks-csv  data/output/corpus_chunks.csv \
        --checkpoint       models/dual_encoder_mpnet/best_model.pt \
        --output-embeddings-npy embeddings/text_chunk_embeddings.npy \
        --output-manifest-csv   embeddings/text_chunk_manifest.csv

Usage mode rapide (apres precompute_text.py) :
    python -m training.export_text_embeddings \
        --precomputed-text-npy      embeddings/precomputed_text.npy \
        --precomputed-text-manifest embeddings/precomputed_text_manifest.csv \
        --text-chunks-csv           data/output/corpus_chunks.csv \
        --checkpoint                models/dual_encoder_mpnet/best_model.pt \
        --output-embeddings-npy     embeddings/text_chunk_embeddings.npy \
        --output-manifest-csv       embeddings/text_chunk_manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .data import load_text_chunks
from .models import DualEncoderModel


def _load_model(checkpoint_path: str) -> DualEncoderModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = DualEncoderModel(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export des embeddings texte projetes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-chunks-csv", required=True,
                        help="Utilise pour la colonne 'text' dans le manifest de sortie.")
    parser.add_argument("--output-embeddings-npy", required=True)
    parser.add_argument("--output-manifest-csv", required=True)
    # Mode rapide
    parser.add_argument("--precomputed-text-npy",
                        help="Embeddings texte pre-calcules (mode rapide).")
    parser.add_argument("--precomputed-text-manifest",
                        help="Manifest des embeddings pre-calcules (colonne chunk_id).")
    # Mode complet
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _export_fast_mode(args, model: DualEncoderModel, chunks: pd.DataFrame) -> None:
    """Mode rapide : applique text_projection sur embeddings pre-calcules."""
    print("[INFO] Mode rapide : application de la text_projection...")
    pre_embs = np.load(args.precomputed_text_npy).astype("float32")
    pre_manifest = pd.read_csv(args.precomputed_text_manifest)

    # Joindre avec le texte original via chunk_id
    merged = pre_manifest.merge(
        chunks[["chunk_id", "document_id", "text"]], on="chunk_id", how="left"
    )

    device = torch.device(args.device)
    model.to(device)

    all_projected = []
    bs = args.batch_size
    bar = tqdm(range(0, len(pre_embs), bs), desc="Projection", unit="batch")
    with torch.no_grad():
        for start in bar:
            batch = torch.tensor(pre_embs[start:start + bs]).to(device)
            projected = model.encode_text_precomputed(batch)
            all_projected.append(projected.cpu().numpy())

    embeddings = np.vstack(all_projected)
    manifest = merged[["chunk_id", "document_id", "text"]].reset_index(drop=True)
    _save(embeddings, manifest, args)


def _export_full_mode(args, model: DualEncoderModel, chunks: pd.DataFrame) -> None:
    """Mode complet : encode le texte avec le text encoder du modele."""
    print("[INFO] Mode complet : encodage texte avec le text encoder...")
    device = torch.device(args.device)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    tokenizer_dir = Path(args.checkpoint).parent / "tokenizer"
    tokenizer_src = tokenizer_dir if tokenizer_dir.exists() else config["text_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
        def __len__(self):
            return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            return {"chunk_id": str(r["chunk_id"]),
                    "document_id": str(r["document_id"]),
                    "text": str(r["text"])}

    def _collate(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt")
        return {**enc,
                "chunk_ids": [b["chunk_id"] for b in batch],
                "document_ids": [b["document_id"] for b in batch],
                "texts": texts}

    loader = DataLoader(_DS(chunks), batch_size=args.batch_size,
                        shuffle=False, collate_fn=_collate, num_workers=0)

    all_embs, manifest_rows = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encodage texte", unit="batch"):
            embs = model.encode_text(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            ).cpu().numpy()
            all_embs.append(embs)
            for cid, did, txt in zip(batch["chunk_ids"], batch["document_ids"], batch["texts"]):
                manifest_rows.append({"chunk_id": cid, "document_id": did, "text": txt})

    embeddings = np.concatenate(all_embs, axis=0)
    manifest = pd.DataFrame(manifest_rows)
    _save(embeddings, manifest, args)


def _save(embeddings: np.ndarray, manifest: pd.DataFrame, args) -> None:
    Path(args.output_embeddings_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_embeddings_npy, embeddings)
    manifest.to_csv(args.output_manifest_csv, index=False)
    print(f"[OK] Embeddings : {args.output_embeddings_npy}  shape={embeddings.shape}")
    print(f"[OK] Manifest   : {args.output_manifest_csv}  ({len(manifest)} lignes)")


def main() -> None:
    args = parse_args()
    model = _load_model(args.checkpoint)
    chunks = load_text_chunks(args.text_chunks_csv)

    fast_mode = bool(args.precomputed_text_npy and args.precomputed_text_manifest)

    if fast_mode:
        if model.text_encoder is not None:
            print("[WARN] Checkpoint en mode complet mais --precomputed-text-npy fourni."
                  " Le mode rapide sera quand meme utilise.")
        _export_fast_mode(args, model, chunks)
    else:
        if model.text_encoder is None:
            raise ValueError(
                "Checkpoint en mode rapide (pas de text encoder). "
                "Fournissez --precomputed-text-npy et --precomputed-text-manifest."
            )
        _export_full_mode(args, model, chunks)


if __name__ == "__main__":
    main()
