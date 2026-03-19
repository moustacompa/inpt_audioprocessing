from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the audio embedding handoff before training."
    )
    parser.add_argument("--audio-manifest-csv", required=True)
    parser.add_argument("--audio-embeddings-npy", required=True)
    parser.add_argument("--pairs-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.audio_manifest_csv)
    embeddings = np.load(args.audio_embeddings_npy)

    if "audio_id" not in manifest.columns:
        raise ValueError("audio_manifest.csv must contain an 'audio_id' column")
    if embeddings.ndim != 2:
        raise ValueError(
            f"audio_embeddings.npy must be a 2D array, got shape {embeddings.shape}"
        )
    if len(manifest) != len(embeddings):
        raise ValueError(
            "audio_manifest.csv row count must match audio_embeddings.npy row count"
        )
    if manifest["audio_id"].duplicated().any():
        duplicate_ids = manifest.loc[manifest["audio_id"].duplicated(), "audio_id"].tolist()
        raise ValueError(f"duplicate audio_id values found: {duplicate_ids[:5]}")
    if np.isnan(embeddings).any():
        raise ValueError("audio_embeddings.npy contains NaN values")

    print(f"manifest_rows={len(manifest)}")
    print(f"embedding_shape={embeddings.shape}")
    print("status=manifest_and_embeddings_ok")

    if not args.pairs_csv:
        return

    pairs = pd.read_csv(args.pairs_csv)
    if "audio_id" not in pairs.columns:
        raise ValueError("pairs.csv must contain an 'audio_id' column")

    manifest_ids = set(manifest["audio_id"].astype(str))
    pair_ids = set(pairs["audio_id"].astype(str))
    missing_ids = sorted(pair_ids.difference(manifest_ids))

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} audio_id values from pairs.csv are missing in audio_manifest.csv. "
            f"Examples: {missing_ids[:5]}"
        )

    print(f"pairs_rows={len(pairs)}")
    print("status=pairs_alignment_ok")


if __name__ == "__main__":
    main()
