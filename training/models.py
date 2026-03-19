from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class DualEncoderModel(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        audio_input_dim: int,
        projection_dim: int = 768,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.text_model_name = text_model_name
        self.audio_input_dim = int(audio_input_dim)
        self.projection_dim = int(projection_dim)
        self.dropout_value = float(dropout)
        self.freeze_text_encoder = bool(freeze_text_encoder)

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        hidden_size = int(self.text_encoder.config.hidden_size)
        self.text_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )
        self.audio_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(audio_input_dim, projection_dim),
        )

        if self.freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        projected = self.text_projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode_audio(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.audio_projection(audio_embeddings)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        speech_embeddings = self.encode_audio(audio_embeddings)
        similarities = speech_embeddings @ text_embeddings.T
        return {
            "speech_embeddings": speech_embeddings,
            "text_embeddings": text_embeddings,
            "similarities": similarities,
        }

    def export_config(self) -> dict[str, Any]:
        return {
            "text_model_name": self.text_model_name,
            "audio_input_dim": self.audio_input_dim,
            "projection_dim": self.projection_dim,
            "dropout": self.dropout_value,
            "freeze_text_encoder": self.freeze_text_encoder,
        }
