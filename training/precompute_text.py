"""
precompute_text.py — Pre-calcul des embeddings texte de base (sans projection).

Lance le text encoder (MPNet ou MiniLM) UNE SEULE FOIS sur tous les chunks,
sauvegarde les vecteurs en .npy. L'entrainement n'a plus besoin de charger
le modele texte : seules les couches de projection sont mises a jour.

Gain typique : 20x a 50x plus rapide en entrainement sur CPU.

Usage :
    python -m training.precompute_text \
        --text-chunks-csv  data/output/corpus_chunks.csv \
        --model-name       sentence-transformers/all-MiniLM-L6-v2 \
        --output-npy       embeddings/precomputed_text.npy \
        --output-manifest  embeddings/precomputed_text_manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from .data import load_text_chunks


class _ChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks.reset_index(drop=True)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        row = self.chunks.iloc[i]
        return {"chunk_id": str(row["chunk_id"]), "text": str(row["text"])}


def _collate(batch, tokenizer, max_length):
    texts = [b["text"] for b in batch]
    chunk_ids = [b["chunk_id"] for b in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return chunk_ids, encoded


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def parse_args():
    p = argparse.ArgumentParser(description="Pre-calcul des embeddings texte.")
    p.add_argument("--text-chunks-csv", required=True)
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modele texte HuggingFace (MiniLM recommande pour la vitesse).",
    )
    p.add_argument("--output-npy", required=True)
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[INFO] Chargement des chunks : {args.text_chunks_csv}")
    chunks = load_text_chunks(args.text_chunks_csv)
    print(f"[INFO] {len(chunks)} chunks a encoder")

    print(f"[INFO] Chargement du modele : {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    dataset = _ChunkDataset(chunks)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate(b, tokenizer, args.max_length),
    )

    all_embeddings = []
    all_chunk_ids = []

    print(f"[INFO] Encodage en cours sur {args.device}...")
    with torch.no_grad():
        for chunk_ids, encoded in tqdm(loader, unit="batch"):
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
            all_embeddings.append(pooled.cpu().float().numpy())
            all_chunk_ids.extend(chunk_ids)

    embeddings = np.vstack(all_embeddings)
    print(f"[INFO] Shape finale : {embeddings.shape}")

    Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, embeddings)
    print(f"[OK]   Embeddings sauvegardes : {args.output_npy}")

    import pandas as pd
    manifest = pd.DataFrame({"chunk_id": all_chunk_ids})
    manifest["embedding_index"] = range(len(manifest))
    manifest.to_csv(args.output_manifest, index=False)
    print(f"[OK]   Manifest sauvegarde   : {args.output_manifest}")
    print(f"[OK]   Dimension des vecteurs : {embeddings.shape[1]}D")


if __name__ == "__main__":
    main()