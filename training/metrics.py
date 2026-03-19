from __future__ import annotations

from typing import Sequence

import torch


def retrieval_metrics(
    speech_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    speech_document_ids: Sequence[str],
    text_document_ids: Sequence[str],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    if len(speech_document_ids) != speech_embeddings.size(0):
        raise ValueError("speech document ids must match the number of speech embeddings")
    if len(text_document_ids) != text_embeddings.size(0):
        raise ValueError("text document ids must match the number of text embeddings")

    similarity = speech_embeddings @ text_embeddings.T
    ranked_indices = torch.argsort(similarity, dim=1, descending=True)

    hits = {int(k): 0 for k in ks}
    reciprocal_ranks = []

    for row_index in range(ranked_indices.size(0)):
        target_document = speech_document_ids[row_index]
        ranking = ranked_indices[row_index].tolist()

        first_relevant_rank = None
        for rank_position, text_index in enumerate(ranking, start=1):
            if text_document_ids[text_index] == target_document:
                first_relevant_rank = rank_position
                break

        if first_relevant_rank is None:
            reciprocal_ranks.append(0.0)
            continue

        reciprocal_ranks.append(1.0 / first_relevant_rank)
        for k in hits:
            if first_relevant_rank <= k:
                hits[k] += 1

    total_queries = max(1, ranked_indices.size(0))
    metrics = {f"Recall@{k}": hits[k] / total_queries for k in hits}
    metrics["MRR"] = sum(reciprocal_ranks) / total_queries
    return metrics
