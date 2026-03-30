from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from .data import (
    AudioEmbeddingStore, TextEmbeddingStore,
    DualEncoderDataset, FastDualEncoderDataset,
    build_collate_fn, fast_collate,
    load_pairs, load_text_chunks,
)
from .losses import contrastive_loss, triplet_loss
from .metrics import retrieval_metrics
from .models import DualEncoderModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the speech-text dual encoder with precomputed audio embeddings."
    )
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--val-pairs-csv")
    parser.add_argument("--audio-manifest-csv", required=True)
    parser.add_argument("--audio-embeddings-npy", required=True)
    parser.add_argument("--output-dir", required=True)
    # --- Mode rapide (recommande sur CPU) ---
    parser.add_argument(
        "--precomputed-text-npy",
        help="Embeddings texte pre-calcules (.npy). Active le mode rapide (pas de text encoder au training).",
    )
    parser.add_argument(
        "--precomputed-text-manifest",
        help="Manifest CSV des embeddings texte pre-calcules (colonne chunk_id requise).",
    )
    # --- Mode complet (necessite GPU pour etre raisonnable) ---
    parser.add_argument("--text-chunks-csv", help="Requis en mode complet uniquement.")
    parser.add_argument(
        "--text-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--projection-dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--loss",
        choices=("contrastive", "triplet"),
        default="contrastive",
    )
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Nombre de steps pour le linear warmup du LR scheduler.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping (0 = désactivé).")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Nombre de workers DataLoader (0 recommandé sur Windows pour éviter les segfaults).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_pair_splits(
    pairs_csv: str,
    val_pairs_csv: str | None,
    train_split: str,
    val_split: str,
):
    pairs = load_pairs(pairs_csv)
    if val_pairs_csv:
        return pairs, load_pairs(val_pairs_csv)

    if "split" not in pairs.columns:
        return pairs, None

    train_pairs = pairs[pairs["split"] == train_split].copy()
    val_pairs = pairs[pairs["split"] == val_split].copy()
    if train_pairs.empty:
        raise ValueError(f"no rows found for train split '{train_split}'")
    if val_pairs.empty:
        val_pairs = None
    return train_pairs, val_pairs


def compute_loss(
    loss_name: str,
    speech_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    if loss_name == "contrastive":
        return contrastive_loss(
            speech_embeddings=speech_embeddings,
            text_embeddings=text_embeddings,
            temperature=temperature,
        )
    return triplet_loss(
        speech_embeddings=speech_embeddings,
        text_embeddings=text_embeddings,
        margin=margin,
    )


def save_checkpoint(
    output_dir: Path,
    model: DualEncoderModel,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    best_metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.export_config(),
            "best_metrics": best_metrics,
            "training_args": vars(args),
        },
        checkpoint_path,
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    with (output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)


@torch.no_grad()
def evaluate(
    model: DualEncoderModel,
    dataloader: DataLoader,
    device: torch.device,
    fast_mode: bool = False,
) -> dict[str, float]:
    model.eval()
    speech_batches = []
    text_batches = []
    chunk_ids: list[str] = []

    for batch in dataloader:
        audio_embeddings = batch["audio_embeddings"].to(device)
        speech_embeddings = model.encode_audio(audio_embeddings)

        if fast_mode:
            text_embeddings = model.encode_text_precomputed(batch["text_embeddings"].to(device))
            chunk_ids.extend(batch["chunk_ids"])
        else:
            text_embeddings = model.encode_text(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            chunk_ids.extend(batch["document_ids"])

        speech_batches.append(speech_embeddings.cpu())
        text_batches.append(text_embeddings.cpu())

    if not speech_batches:
        return {"Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}

    return retrieval_metrics(
        speech_embeddings=torch.cat(speech_batches, dim=0),
        text_embeddings=torch.cat(text_batches, dim=0),
        speech_document_ids=chunk_ids,
        text_document_ids=chunk_ids,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # Determine le mode : rapide (embeddings pre-calcules) ou complet (text encoder)
    fast_mode = bool(args.precomputed_text_npy and args.precomputed_text_manifest)
    if fast_mode:
        print("[INFO] Mode RAPIDE : embeddings texte pre-calcules (pas de text encoder au training)")
    else:
        print("[INFO] Mode COMPLET : text encoder charge a chaque batch (lent sur CPU)")
        if not args.text_chunks_csv:
            raise ValueError("--text-chunks-csv requis en mode complet (ou utiliser --precomputed-text-npy).")

    audio_store = AudioEmbeddingStore.load(
        manifest_path=args.audio_manifest_csv,
        embeddings_path=args.audio_embeddings_npy,
    )
    train_pairs, val_pairs = prepare_pair_splits(
        pairs_csv=args.pairs_csv,
        val_pairs_csv=args.val_pairs_csv,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    if fast_mode:
        text_store = TextEmbeddingStore.load(
            manifest_path=args.precomputed_text_manifest,
            embeddings_path=args.precomputed_text_npy,
        )
        tokenizer = None
        collate_fn = fast_collate
        train_dataset = FastDualEncoderDataset(train_pairs, audio_store, text_store)
    else:
        text_chunks = load_text_chunks(args.text_chunks_csv)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
        collate_fn = build_collate_fn(tokenizer, max_length=args.max_length)
        train_dataset = DualEncoderDataset(train_pairs, text_chunks, audio_store)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = None
    if val_pairs is not None and not val_pairs.empty:
        if fast_mode:
            val_dataset = FastDualEncoderDataset(val_pairs, audio_store, text_store)
            val_collate = fast_collate
        else:
            val_dataset = DualEncoderDataset(val_pairs, text_chunks, audio_store)
            val_collate = collate_fn
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    model = DualEncoderModel(
        audio_input_dim=audio_store.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        text_model_name=None if fast_mode else args.text_model_name,
        text_input_dim=text_store.embedding_dim if fast_mode else None,
        freeze_text_encoder=args.freeze_text_encoder,
    ).to(device)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    # Linear warmup puis LR constant (conformément au README)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = min(args.warmup_steps, total_steps)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    decay_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-2,
        total_iters=max(1, total_steps - warmup_steps),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    best_score = float("-inf")
    best_metrics: dict[str, float] = {"Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}

    n_batches = len(train_loader)
    print(
        f"\nDébut de l'entraînement : {args.epochs} époques | "
        f"{len(train_loader.dataset)} échantillons | "
        f"{n_batches} batches/époque | "
        f"device={args.device}\n"
    )

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Entraînement", unit="époque")

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"  Époque {epoch}/{args.epochs}",
            unit="batch",
            leave=False,
        )

        for batch in batch_bar:
            optimizer.zero_grad()

            audio_embeddings = batch["audio_embeddings"].to(device)

            if fast_mode:
                outputs = model(
                    audio_embeddings=audio_embeddings,
                    text_embeddings=batch["text_embeddings"].to(device),
                )
            else:
                outputs = model(
                    audio_embeddings=audio_embeddings,
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            loss = compute_loss(
                loss_name=args.loss,
                speech_embeddings=outputs["speech_embeddings"],
                text_embeddings=outputs["text_embeddings"],
                temperature=args.temperature,
                margin=args.margin,
            )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

            batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        average_loss = running_loss / max(1, n_batches)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, fast_mode=fast_mode)
            epoch_bar.set_postfix(
                loss=f"{average_loss:.4f}",
                R5=f"{metrics['Recall@5']:.3f}",
                R10=f"{metrics['Recall@10']:.3f}",
                MRR=f"{metrics['MRR']:.3f}",
            )
            tqdm.write(
                f"[Époque {epoch:>3}/{args.epochs}] "
                f"loss={average_loss:.4f} | "
                f"Recall@5={metrics['Recall@5']:.4f} | "
                f"Recall@10={metrics['Recall@10']:.4f} | "
                f"MRR={metrics['MRR']:.4f}"
                + (" ← meilleur" if metrics["MRR"] > best_score else "")
            )
            if metrics["MRR"] > best_score:
                best_score = metrics["MRR"]
                best_metrics = metrics
                save_checkpoint(output_dir, model, tokenizer, args, best_metrics)
        else:
            epoch_bar.set_postfix(loss=f"{average_loss:.4f}")
            tqdm.write(f"[Époque {epoch:>3}/{args.epochs}] loss={average_loss:.4f}")

    if val_loader is None:
        save_checkpoint(output_dir, model, tokenizer, args, best_metrics)

    print(f"\nEntraînement terminé. Meilleur MRR={best_score:.4f} → {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
