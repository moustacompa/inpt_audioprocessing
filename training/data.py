from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TEXT_REQUIRED_COLUMNS = {"chunk_id", "document_id", "text"}
PAIR_MIN_COLUMNS = {"audio_id"}
MANIFEST_MIN_COLUMNS = {"audio_id"}


def _derive_document_id(df: pd.DataFrame) -> pd.DataFrame:
    """Dérive document_id depuis chunk_id si absent (ex: doc00227_chunk031 → doc00227)."""
    if "document_id" not in df.columns and "chunk_id" in df.columns:
        df = df.copy()
        df["document_id"] = df["chunk_id"].str.replace(r"_chunk\d+$", "", regex=True)
    return df


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_text_chunks(path: str | Path) -> pd.DataFrame:
    chunks = _read_csv(path)
    # Toujours dériver document_id depuis chunk_id (garantit le type str et la cohérence
    # avec les paires, même si doc_idx existe mais est numérique)
    chunks = _derive_document_id(chunks)
    missing = TEXT_REQUIRED_COLUMNS.difference(chunks.columns)
    if missing:
        raise ValueError(
            f"text chunks file is missing columns: {sorted(missing)}"
        )
    return chunks.copy()


def load_pairs(path: str | Path) -> pd.DataFrame:
    pairs = _read_csv(path)
    # Normalise audio_file → audio_id (supprime l'extension .wav si présente)
    if "audio_id" not in pairs.columns and "audio_file" in pairs.columns:
        pairs = pairs.rename(columns={"audio_file": "audio_id"})
        pairs["audio_id"] = pairs["audio_id"].str.replace(r"\.wav$", "", regex=True)
    pairs = _derive_document_id(pairs)
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


# ---------------------------------------------------------------------------
# Mode rapide : embeddings texte pre-calcules (pas de text encoder au training)
# ---------------------------------------------------------------------------

@dataclass
class TextEmbeddingStore:
    """Embeddings texte pre-calcules (symetrique a AudioEmbeddingStore)."""
    manifest: pd.DataFrame
    embeddings: np.ndarray

    @classmethod
    def load(cls, manifest_path: str | Path, embeddings_path: str | Path) -> "TextEmbeddingStore":
        manifest = _read_csv(Path(manifest_path))
        if "chunk_id" not in manifest.columns:
            raise ValueError("text embedding manifest must contain a 'chunk_id' column")
        embeddings = np.load(embeddings_path)
        if embeddings.ndim != 2:
            raise ValueError(f"expected 2D array for text embeddings, got {embeddings.shape}")
        if len(manifest) != len(embeddings):
            raise ValueError("text manifest rows must match text embeddings rows")
        manifest = manifest.copy()
        if "embedding_index" not in manifest.columns:
            manifest["embedding_index"] = np.arange(len(manifest), dtype=np.int64)
        return cls(manifest=manifest, embeddings=embeddings)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def attach_to_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        merged = pairs.merge(
            self.manifest[["chunk_id", "embedding_index"]].rename(
                columns={"embedding_index": "text_embedding_index"}
            ),
            on="chunk_id",
            how="inner",
        )
        if merged.empty:
            raise ValueError("no pairs matched the text embedding manifest on chunk_id")
        return merged


class FastDualEncoderDataset(Dataset):
    """Dataset sans tokenisation : utilise des embeddings audio ET texte pre-calcules.
    Reduit le temps d'entrainement de 20x a 50x sur CPU.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        audio_store: AudioEmbeddingStore,
        text_store: TextEmbeddingStore,
    ) -> None:
        merged = audio_store.attach_to_pairs(pairs)
        merged = text_store.attach_to_pairs(merged)
        self.samples = merged.reset_index(drop=True)
        self.audio_embeddings = audio_store.embeddings
        self.text_embeddings = text_store.embeddings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        return {
            "audio_id": row["audio_id"],
            "chunk_id": str(row["chunk_id"]),
            "audio_embedding": torch.tensor(
                self.audio_embeddings[int(row["embedding_index"])], dtype=torch.float32
            ),
            "text_embedding": torch.tensor(
                self.text_embeddings[int(row["text_embedding_index"])], dtype=torch.float32
            ),
        }


def fast_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio_embeddings": torch.stack([b["audio_embedding"] for b in batch]),
        "text_embeddings": torch.stack([b["text_embedding"] for b in batch]),
        "audio_ids": [b["audio_id"] for b in batch],
        "chunk_ids": [b["chunk_id"] for b in batch],
    }
