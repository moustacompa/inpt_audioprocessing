"""
speech_encoder.py — P2: Speech AI Engineer
===========================================
Encodeur audio basé sur Wav2Vec2 pour le système Speech-to-Retrieval.
Prend un fichier .wav brut et produit un vecteur normalisé L2 de 768 dims,
SANS transcription intermédiaire (pas d'ASR).

Usage:
    from src.speech_encoder import speech_to_embedding, SpeechEncoder

    # Inférence directe (fichier → vecteur numpy)
    emb = speech_to_embedding("query.wav")   # → np.ndarray (768,)

    # Classe PyTorch (pour l'entraînement P3)
    encoder = SpeechEncoder(frozen=True).to(device)
    emb = encoder(input_values)              # → Tensor [B, 768]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16_000
EMBEDDING_DIM = 768  # dimension de sortie de wav2vec2-base


# ---------------------------------------------------------------------------
# Classe PyTorch : utilisée par P3 pendant l'entraînement du dual encoder
# ---------------------------------------------------------------------------

class SpeechEncoder(nn.Module):
    """Encodeur audio Wav2Vec2 pour le dual-encoder S2R.

    Paramètres
    ----------
    model_name : str
        Identifiant HuggingFace du modèle Wav2Vec2.
    frozen : bool
        Si True, gèle les poids Wav2Vec2 (seule la couche de projection
        du dual encoder sera entraînée — recommandé pour P3).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if frozen:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return int(self.wav2vec2.config.hidden_size)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode un batch de waveforms en vecteurs L2-normalisés.

        Paramètres
        ----------
        input_values : Tensor [B, T]
            Waveforms bruts à 16 kHz.
        attention_mask : Tensor [B, T], optionnel
            Masque pour les séquences paddées.

        Retourne
        --------
        Tensor [B, 768]
            Embeddings L2-normalisés.
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # [B, T', 768]

        if attention_mask is not None:
            # Propager le masque jusqu'à la sortie du feature extractor
            out_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1)
            )
            seq_len = hidden.size(1)
            mask = torch.zeros(
                (hidden.size(0), seq_len),
                dtype=torch.float32,
                device=hidden.device,
            )
            for i, length in enumerate(out_lengths):
                mask[i, : int(length)] = 1.0
            mask = mask.unsqueeze(-1)  # [B, T', 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)  # [B, 768]

        return F.normalize(pooled, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Utilitaires d'inférence standalone
# ---------------------------------------------------------------------------

def _load_wav(audio_path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Charge un fichier audio et le convertit en waveform mono 16 kHz.

    Retourne
    --------
    Tensor [T]
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:  # stéréo → mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)  # [T]


def speech_to_embedding(
    audio_path: str,
    model_name: str = MODEL_NAME,
    device: str = "cpu",
) -> np.ndarray:
    """Convertit un fichier .wav en vecteur d'embedding L2-normalisé.

    C'est la fonction principale du système S2R côté requête :
    audio brut → embedding, SANS ASR, SANS transcription.

    Paramètres
    ----------
    audio_path : str
        Chemin vers un fichier .wav (16 kHz mono recommandé).
    model_name : str
        Modèle Wav2Vec2 à utiliser.
    device : str
        Device torch ('cpu' ou 'cuda').

    Retourne
    --------
    np.ndarray de forme (768,)
    """
    encoder = SpeechEncoder(model_name=model_name, frozen=True).to(device)
    encoder.eval()

    waveform = _load_wav(audio_path).to(device)

    with torch.no_grad():
        emb = encoder(waveform.unsqueeze(0))  # [1, 768]

    return emb.squeeze(0).cpu().numpy()  # (768,)


def batch_speech_to_embeddings(
    audio_paths: list[str],
    model_name: str = MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Encode une liste de fichiers audio en embeddings (avec batching).

    Gère le padding pour les séquences de longueurs différentes.

    Retourne
    --------
    np.ndarray de forme (N, 768)
    """
    encoder = SpeechEncoder(model_name=model_name, frozen=True).to(device)
    encoder.eval()

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        waveforms = [_load_wav(p) for p in batch_paths]

        # Padding à la longueur max du batch
        max_len = max(w.size(0) for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len, device=device)
        mask = torch.zeros(len(waveforms), max_len, dtype=torch.long, device=device)

        for i, w in enumerate(waveforms):
            padded[i, : w.size(0)] = w.to(device)
            mask[i, : w.size(0)] = 1

        with torch.no_grad():
            embs = encoder(padded, attention_mask=mask)  # [B, 768]

        all_embeddings.append(embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def export_to_onnx(output_path: str, model_name: str = MODEL_NAME) -> None:
    """Exporte l'encodeur en ONNX pour une inférence plus rapide (optionnel P4).

    Paramètres
    ----------
    output_path : str
        Chemin du fichier .onnx à créer.
    model_name : str
        Modèle Wav2Vec2 source.
    """
    encoder = SpeechEncoder(model_name=model_name, frozen=True).eval()
    dummy = torch.randn(1, SAMPLE_RATE * 5)  # 5 secondes d'audio fictif
    torch.onnx.export(
        encoder,
        dummy,
        output_path,
        input_names=["input_values"],
        output_names=["embeddings"],
        dynamic_axes={"input_values": {0: "batch_size", 1: "time"}},
        opset_version=14,
    )
    print(f"SpeechEncoder exporté → {output_path}")
