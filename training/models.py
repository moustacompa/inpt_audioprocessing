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
        audio_input_dim: int,
        projection_dim: int = 256,
        dropout: float = 0.1,
        # Mode complet : text_model_name charge le text encoder
        text_model_name: str | None = None,
        freeze_text_encoder: bool = False,
        # Mode rapide : text_input_dim si embeddings pre-calcules (pas de text encoder)
        text_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.text_model_name = text_model_name
        self.audio_input_dim = int(audio_input_dim)
        self.projection_dim = int(projection_dim)
        self.dropout_value = float(dropout)
        self.freeze_text_encoder = bool(freeze_text_encoder)

        if text_model_name is not None:
            # Mode complet : text encoder charge depuis HuggingFace
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            hidden_size = int(self.text_encoder.config.hidden_size)
            if freeze_text_encoder:
                for parameter in self.text_encoder.parameters():
                    parameter.requires_grad = False
        elif text_input_dim is not None:
            # Mode rapide : pas de text encoder, projection directe
            self.text_encoder = None
            hidden_size = int(text_input_dim)
        else:
            raise ValueError("Fournir text_model_name (mode complet) ou text_input_dim (mode rapide).")

        self.text_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )
        self.audio_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(audio_input_dim, projection_dim),
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mode complet uniquement (text encoder charge)."""
        if self.text_encoder is None:
            raise RuntimeError("encode_text() non disponible en mode rapide. Utiliser encode_text_precomputed().")
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(self.text_projection(pooled), p=2, dim=-1)

    def encode_text_precomputed(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Mode rapide : projette des embeddings texte pre-calcules."""
        return F.normalize(self.text_projection(text_embeddings), p=2, dim=-1)

    def encode_audio(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.audio_projection(audio_embeddings), p=2, dim=-1)

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        speech_emb = self.encode_audio(audio_embeddings)
        if text_embeddings is not None:
            text_emb = self.encode_text_precomputed(text_embeddings)
        else:
            text_emb = self.encode_text(input_ids, attention_mask)
        return {
            "speech_embeddings": speech_emb,
            "text_embeddings": text_emb,
            "similarities": speech_emb @ text_emb.T,
        }

    def export_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "audio_input_dim": self.audio_input_dim,
            "projection_dim": self.projection_dim,
            "dropout": self.dropout_value,
            "freeze_text_encoder": self.freeze_text_encoder,
            "text_model_name": self.text_model_name,
        }
        if self.text_encoder is None:
            # Mode rapide : sauvegarder la dim d'entrée de la text_projection
            cfg["text_input_dim"] = int(self.text_projection[1].in_features)
        return cfg
