from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TEXT_REQUIRED_COLUMNS = {"chunk_id", "document_id", "text"}
PAIR_MIN_COLUMNS = {"audio_id", "document_id"}
MANIFEST_MIN_COLUMNS = {"audio_id"}


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_text_chunks(path: str | Path) -> pd.DataFrame:
    chunks = _read_csv(path)
    missing = TEXT_REQUIRED_COLUMNS.difference(chunks.columns)
    if missing:
        raise ValueError(
            f"text chunks file is missing columns: {sorted(missing)}"
        )
    return chunks.copy()


def load_pairs(path: str | Path) -> pd.DataFrame:
    pairs = _read_csv(path)
    missing = PAIR_MIN_COLUMNS.difference(pairs.columns)
    if missing:
        raise ValueError(f"pairs file is missing columns: {sorted(missing)}")
    return pairs.copy()


@dataclass
class AudioEmbeddingStore:
    manifest_path: Path
    embeddings_path: Path
    manifest: pd.DataFrame
    embeddings: np.ndarray

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        embeddings_path: str | Path,
    ) -> "AudioEmbeddingStore":
        manifest_csv = Path(manifest_path)
        embeddings_npy = Path(embeddings_path)
        manifest = _read_csv(manifest_csv)
        missing = MANIFEST_MIN_COLUMNS.difference(manifest.columns)
        if missing:
            raise ValueError(
                f"audio embedding manifest is missing columns: {sorted(missing)}"
            )

        embeddings = np.load(embeddings_npy)
        if embeddings.ndim != 2:
            raise ValueError(
                f"expected a 2D numpy array for audio embeddings, got shape {embeddings.shape}"
            )
        if len(manifest) != len(embeddings):
            raise ValueError(
                "audio embedding manifest rows must match the number of embedding rows"
            )

        manifest = manifest.copy()
        if "embedding_index" not in manifest.columns:
            manifest["embedding_index"] = np.arange(len(manifest), dtype=np.int64)

        return cls(
            manifest_path=manifest_csv,
            embeddings_path=embeddings_npy,
            manifest=manifest,
            embeddings=embeddings,
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def attach_to_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        merged = pairs.merge(
            self.manifest[["audio_id", "embedding_index"]],
            on="audio_id",
            how="inner",
        )
        if merged.empty:
            raise ValueError(
                "no training pairs matched the provided audio embedding manifest"
            )
        return merged


def resolve_pairs_with_text(
    pairs: pd.DataFrame,
    text_chunks: pd.DataFrame,
) -> pd.DataFrame:
    merged = pairs.copy()

    if "chunk_id" not in merged.columns:
        first_chunks = (
            text_chunks.sort_values(["document_id", "chunk_id"])
            .drop_duplicates(subset=["document_id"], keep="first")
            [["document_id", "chunk_id", "text"]]
        )
        merged = merged.merge(first_chunks, on="document_id", how="inner")
    else:
        merged = merged.merge(
            text_chunks[["chunk_id", "document_id", "text"]],
            on=["chunk_id", "document_id"],
            how="inner",
        )

    if merged.empty:
        raise ValueError(
            "no text chunks matched the training pairs. Check document_id/chunk_id alignment."
        )
    return merged


class DualEncoderDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        text_chunks: pd.DataFrame,
        audio_store: AudioEmbeddingStore,
    ) -> None:
        merged_pairs = resolve_pairs_with_text(pairs, text_chunks)
        merged_pairs = audio_store.attach_to_pairs(merged_pairs)
        self.samples = merged_pairs.reset_index(drop=True)
        self.audio_embeddings = audio_store.embeddings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        embedding_index = int(row["embedding_index"])
        audio_embedding = torch.tensor(
            self.audio_embeddings[embedding_index],
            dtype=torch.float32,
        )
        return {
            "audio_id": row["audio_id"],
            "document_id": str(row["document_id"]),
            "chunk_id": str(row["chunk_id"]),
            "text": str(row["text"]),
            "audio_embedding": audio_embedding,
        }


def build_collate_fn(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "audio_embeddings": torch.stack(
                [item["audio_embedding"] for item in batch]
            ),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "audio_ids": [item["audio_id"] for item in batch],
            "document_ids": [item["document_id"] for item in batch],
            "chunk_ids": [item["chunk_id"] for item in batch],
            "texts": texts,
        }

    return collate
