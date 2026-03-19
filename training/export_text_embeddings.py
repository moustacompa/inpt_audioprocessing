from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import load_text_chunks
from .models import DualEncoderModel


class TextChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks: pd.DataFrame) -> None:
        self.chunks = chunks.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.chunks.iloc[index]
        return {
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "text": str(row["text"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export text chunk embeddings for retrieval."
    )
    parser.add_argument("--text-chunks-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-embeddings-npy", required=True)
    parser.add_argument("--output-manifest-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collate_fn(tokenizer, max_length):
    def collate(batch):
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "chunk_ids": [item["chunk_id"] for item in batch],
            "document_ids": [item["document_id"] for item in batch],
            "texts": texts,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return collate


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]

    model = DualEncoderModel(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    tokenizer_dir = Path(args.checkpoint).resolve().parent / "tokenizer"
    tokenizer_source = tokenizer_dir if tokenizer_dir.exists() else config["text_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    chunks = load_text_chunks(args.text_chunks_csv)
    dataset = TextChunkDataset(chunks)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, args.max_length),
    )

    all_embeddings = []
    manifest_rows = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_embeddings = model.encode_text(input_ids, attention_mask).cpu().numpy()
            all_embeddings.append(text_embeddings)

            for chunk_id, document_id, text in zip(
                batch["chunk_ids"],
                batch["document_ids"],
                batch["texts"],
            ):
                manifest_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "text": text,
                    }
                )

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(args.output_embeddings_npy, embeddings)
    pd.DataFrame(manifest_rows).to_csv(args.output_manifest_csv, index=False)


if __name__ == "__main__":
    main()
