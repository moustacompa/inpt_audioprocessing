from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    speech_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = (speech_embeddings @ text_embeddings.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_speech = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels)
    return (loss_speech + loss_text) / 2.0


def triplet_loss(
    speech_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    similarity = speech_embeddings @ text_embeddings.T
    diagonal = similarity.diag()
    negative_mask = torch.eye(
        similarity.size(0),
        device=similarity.device,
        dtype=torch.bool,
    )

    hardest_text_negative = similarity.masked_fill(
        negative_mask,
        float("-inf"),
    ).max(dim=1).values
    hardest_speech_negative = similarity.T.masked_fill(
        negative_mask,
        float("-inf"),
    ).max(dim=1).values

    positive_distance = 1.0 - diagonal
    negative_text_distance = 1.0 - hardest_text_negative
    negative_speech_distance = 1.0 - hardest_speech_negative

    speech_anchor_loss = F.relu(
        positive_distance - negative_text_distance + margin
    )
    text_anchor_loss = F.relu(
        positive_distance - negative_speech_distance + margin
    )
    return (speech_anchor_loss.mean() + text_anchor_loss.mean()) / 2.0
