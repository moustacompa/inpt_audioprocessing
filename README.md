# Speech-to-Retrieval (S2R) — INPT Deep Learning Project

Système de **recherche documentaire directement depuis la voix**, sans transcription ASR intermédiaire. Une requête audio est encodée en vecteur puis comparée à un index FAISS de documents texte pour retourner les passages les plus pertinents.

> Idéal pour des contextes multilingues (anglais/français) où la transcription automatique introduit trop d'erreurs.

---

## Architecture du pipeline

```
Requête audio (.wav)
        │
        ▼
  [Wav2Vec2 Encoder]          ← facebook/wav2vec2-base-960h
        │ mean pooling + L2 norm
        ▼
  [Audio Projection]          ← couche linéaire du Dual Encoder (P3)
        │ vecteur 768D
        ▼
  [FAISS IndexFlatIP]         ← Inner Product = cosine similarity
        │ top-k indices
        ▼
  Documents pertinents

Corpus texte (.csv / .txt)
        │
        ▼
  [Text Encoder (MPNet)]      ← sentence-transformers/all-mpnet-base-v2
        │ mean pooling + L2 norm
        ▼
  [Text Projection]           ← couche linéaire du Dual Encoder (P3)
        │ vecteur 768D
        ▼
  [FAISS IndexFlatIP]         ← index pré-calculé (export_text_embeddings.py)
```

Les deux espaces sont **alignés pendant l'entraînement** via une **contrastive loss (InfoNCE)** ou une **triplet loss**.

---

## Structure du projet

```
inpt_audioprocessing/
│
├── data/
│   ├── audio_clean/          # 4 985 fichiers .wav (corpus audio LibriSpeech)
│   ├── audio_queries/        # 11 fichiers .wav (requêtes de test)
│   ├── chunks/               # 6 744 fichiers .txt (chunks texte bruts)
│   ├── documents/            # corpus texte source (ss-corpus-fr.tsv)
│   └── output/               # CSVs générés par P1/P2
│       ├── audio_manifest.csv
│       ├── corpus_chunks.csv
│       ├── pairs.csv
│       ├── pairs_train.csv / pairs_val.csv / pairs_test.csv
│       └── pairs_*_with_embeddings.csv
│
├── embeddings/
│   ├── audio_embeddings.npy          # embeddings Wav2Vec2 pré-calculés (P2)
│   ├── audio_embeddings_index.csv    # manifest aligné avec audio_embeddings.npy
│   └── text_embeddings.npy           # embeddings texte MiniLM (prototype P1)
│
├── notebooks/
│   ├── P1_Data_Engineer_v3.ipynb     # Préparation des données
│   ├── P2_Speech_AI_Engineer_v2_FIXED.ipynb  # Encodage audio + embeddings
│   └── MAMADOU_README.md             # Documentation P1/P2 détaillée
│
├── src/
│   └── speech_encoder.py     # SpeechEncoder (Wav2Vec2), speech_to_embedding()
│
├── training/
│   ├── models.py             # DualEncoderModel (audio + text projections)
│   ├── losses.py             # contrastive_loss (InfoNCE), triplet_loss
│   ├── metrics.py            # Recall@5, Recall@10, MRR
│   ├── data.py               # Dataset, AudioEmbeddingStore, chargement CSV
│   ├── train_dual_encoder.py # Script d'entraînement principal
│   ├── export_text_embeddings.py  # Export embeddings texte post-entraînement
│   ├── validate_handoff.py   # Vérification de la compatibilité des données
│   └── PERSON2_HANDOFF.md    # Contrat d'interface P2 → P3
│
├── utils/
│   ├── create_embeddings.py  # Génération des embeddings texte et audio
│   ├── faiss_index.py        # Construction et gestion de l'index FAISS
│   ├── search.py             # Recherche par texte ou par audio
│   └── test.py               # Tests du pipeline de recherche
│
├── api/
│   └── main.py               # API FastAPI (endpoints /search, /search/file, /health)
│
├── demo/
│   └── app.py                # Interface Gradio (microphone + upload)
│
├── inference.py              # Script CLI end-to-end (audio → top-k docs)
├── requirements.txt
└── ss-corpus-fr.tsv          # Corpus texte francophone source
```

---

## Installation

**Prérequis :** Python 3.10+, pip

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd inpt_audioprocessing

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

> **GPU** : Si vous disposez d'un GPU NVIDIA, installez PyTorch avec support CUDA depuis [pytorch.org](https://pytorch.org) avant d'exécuter pip install.

---

## Procédure d'exécution complète

### Étape 1 — Vérifier les données (P1/P2 déjà réalisés)

Les données ont été préparées dans les notebooks. Vérifiez que les fichiers requis sont présents :

```bash
python -m training.validate_handoff \
  --audio-manifest-csv  data/output/audio_manifest.csv \
  --audio-embeddings    embeddings/audio_embeddings.npy \
  --text-chunks-csv     data/output/corpus_chunks.csv \
  --pairs-csv           data/output/pairs_train.csv
```

Si tout est valide, vous pouvez passer directement à l'étape 3.

---

### Étape 2 — Régénérer les embeddings audio (optionnel)

Si vous souhaitez recalculer les embeddings audio depuis les fichiers `.wav` :

```bash
python utils/create_embeddings.py \
  --mode audio \
  --audio-folder data/audio_clean \
  --output-path  embeddings/audio_embeddings.npy \
  --manifest-path embeddings/audio_embeddings_index.csv
```

Pour régénérer les embeddings texte du corpus (prototype MiniLM) :

```bash
python utils/create_embeddings.py --mode text  --input-csv   data/output/corpus_chunks.csv --output-path embeddings/text_embeddings.npy
```

---

### Étape 3 — Entraîner le Dual Encoder (P3)

L'entraînement aligne l'espace audio (Wav2Vec2 768D) et l'espace texte (MPNet 768D) dans un espace commun de dimension `--projection-dim`.

```bash
python -m training.train_dual_encoder   --text-chunks-csv       data/output/corpus_chunks.csv --pairs-csv             data/output/pairs_train.csv --val-pairs-csv         data/output/pairs_val.csv  --audio-manifest-csv    embeddings/audio_embeddings_index.csv --audio-embeddings-npy  embeddings/audio_embeddings.npy --output-dir            models/dual_encoder_mpnet   --text-model-name       sentence-transformers/all-mpnet-base-v2 \
  --projection-dim        768 \
  --epochs                20 \
  --batch-size            16 \
  --learning-rate         2e-5 \
  --warmup-steps          100 \
  --max-grad-norm         1.0 \
  --loss                  contrastive \
  --temperature           0.07
```

**Options notables :**

| Argument | Défaut | Description |
|---|---|---|
| `--loss` | `contrastive` | `contrastive` (InfoNCE) ou `triplet` |
| `--projection-dim` | `768` | Dimension de l'espace partagé |
| `--warmup-steps` | `100` | Steps de warmup linéaire du LR |
| `--max-grad-norm` | `1.0` | Gradient clipping (`0` = désactivé) |
| `--freeze-text-encoder` | off | Geler le text encoder pendant l'entraînement |
| `--device` | auto | `cpu` ou `cuda` |

Le modèle est sauvegardé dans `models/dual_encoder_mpnet/best_model.pt`.

---

### Étape 4 — Exporter les embeddings texte (après entraînement)

Encode tous les chunks texte du corpus dans l'espace partagé du dual encoder :

```bash
python -m training.export_text_embeddings \
  --text-chunks-csv         data/output/corpus_chunks.csv \
  --checkpoint              models/dual_encoder_mpnet/best_model.pt \
  --output-embeddings-npy   embeddings/text_chunk_embeddings.npy \
  --output-manifest-csv     embeddings/text_chunk_manifest.csv \
  --batch-size              32
```

---

### Étape 5 — Inférence en ligne de commande

```bash
# Avec le dual encoder entraîné (recommandé)
python inference.py \
  --audio          data/audio_queries/spontaneous-speech-fr-20300.wav \
  --checkpoint     models/dual_encoder_mpnet/best_model.pt \
  --text-embeddings embeddings/text_chunk_embeddings.npy \
  --manifest        embeddings/text_chunk_manifest.csv \
  --k               5

# Mode prototype (sans dual encoder — espaces non alignés, résultats indicatifs)
python inference.py \
  --audio          data/audio_queries/spontaneous-speech-fr-20300.wav \
  --text-embeddings embeddings/audio_embeddings.npy \
  --manifest        embeddings/audio_embeddings_index.csv \
  --k               5
```

---

### Étape 6 — Interface Gradio (démo interactive)

Lance une interface web avec entrée microphone ou fichier audio :

```bash
# Mode standard
python demo/app.py

# Avec le dual encoder entraîné
python demo/app.py --checkpoint models/dual_encoder_mpnet/best_model.pt

# Avec lien public partageable (Gradio share)
python demo/app.py --checkpoint models/dual_encoder_mpnet/best_model.pt --share

# Sur un port personnalisé
python demo/app.py --port 7861
```

Ouvrir [http://localhost:7860](http://localhost:7860) dans le navigateur.

---

### Étape 7 — API REST FastAPI

Lance une API HTTP pour intégrer S2R dans d'autres applications :

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Variables d'environnement (optionnelles) :**

```bash
export S2R_CHECKPOINT="models/dual_encoder_mpnet/best_model.pt"
export S2R_TEXT_EMBEDDINGS="embeddings/text_chunk_embeddings.npy"
export S2R_MANIFEST="embeddings/text_chunk_manifest.csv"
export S2R_DEVICE="cuda"   # ou "cpu"
```

**Endpoints disponibles :**

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | État de l'API (dual encoder chargé, taille de l'index) |
| `POST` | `/search` | Audio encodé en base64 → top-k documents |
| `POST` | `/search/file` | Upload d'un fichier `.wav` → top-k documents |

**Exemples d'appels :**

```bash
# Health check
curl http://localhost:8000/health

# Recherche par fichier .wav
curl -X POST http://localhost:8000/search/file \
  -F "file=@data/audio_queries/spontaneous-speech-fr-20300.wav" \
  -F "k=5"

# Recherche par audio base64
AUDIO_B64=$(base64 -w 0 data/audio_queries/spontaneous-speech-fr-20300.wav)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$AUDIO_B64\", \"k\": 5}"
```

Documentation interactive disponible sur [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Tests rapides

```bash
# Tester la recherche audio (S2R)
python utils/test.py --mode audio \
  --audio data/audio_queries/spontaneous-speech-fr-20300.wav

# Tester la recherche texte (prototype)
python utils/test.py --mode text \
  --query "Dix-huit heures, tout rond."
```

---

## Description des modules

### `src/speech_encoder.py`

Encodeur speech sans ASR basé sur Wav2Vec2 :

- `SpeechEncoder` — module PyTorch utilisé pendant l'entraînement (P3)
- `speech_to_embedding(audio_path)` → `np.ndarray (768,)` — inférence simple
- `batch_speech_to_embeddings(audio_paths)` → `np.ndarray (N, 768)` — traitement batch avec padding

### `training/models.py`

Architecture du Dual Encoder :

- `SpeechProjection` — projette l'embedding audio 768D vers l'espace commun
- `TextProjection` — projette l'embedding texte MPNet vers l'espace commun
- `DualEncoderModel` — encapsule les deux projections, expose `encode_audio()` et `encode_text()`

### `training/losses.py`

- `contrastive_loss(audio_emb, text_emb, temperature)` — InfoNCE / NT-Xent loss
- `triplet_loss(anchor, positive, negative, margin)` — Triplet loss avec marge

### `training/metrics.py`

- `retrieval_metrics(audio_emb, text_emb)` — calcule Recall@5, Recall@10 et MRR sur le batch de validation

---

## État des données (P1/P2)

| Fichier | Contenu | Taille |
|---|---|---|
| `data/audio_clean/` | Corpus audio LibriSpeech nettoyé | 4 985 fichiers `.wav` |
| `data/chunks/` | Chunks texte bruts | 6 744 fichiers `.txt` |
| `data/output/audio_manifest.csv` | Index audio (audio_id, filepath, text, SNR) | 4 985 lignes |
| `data/output/corpus_chunks.csv` | Chunks texte (chunk_id, document_id, text) | ~6 744 lignes |
| `data/output/pairs_train.csv` | Paires audio–texte pour l'entraînement | 3 988 lignes |
| `data/output/pairs_val.csv` | Paires de validation | 498 lignes |
| `data/output/pairs_test.csv` | Paires de test | 499 lignes |
| `embeddings/audio_embeddings.npy` | Embeddings Wav2Vec2 pré-calculés | shape (4985, 768) |

---

## Notes importantes

- **Dimension mismatch (prototype)** : Les embeddings texte MiniLM (`text_embeddings.npy`) sont en 384D, les embeddings audio Wav2Vec2 sont en 768D. Les espaces ne sont pas alignés avant l'entraînement du dual encoder. Le pipeline le détecte et affiche un message explicite.
- **Mode prototype** : Sans dual encoder entraîné, la démo et l'API fonctionnent en mode dégradé (similarité dans l'espace Wav2Vec2 brut, non aligné avec le texte).
- **Après entraînement** : Utiliser `export_text_embeddings.py` pour indexer le corpus texte dans l'espace partagé du dual encoder avant de lancer la démo ou l'API.
