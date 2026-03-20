# 🎙️ Speech-to-Retrieval System (SRS)
### Projet P3 — INPT | Deep Learning

> Système de recherche documentaire directement depuis la voix, sans transcription intermédiaire.  
> Pipeline : **Audio → Embedding → FAISS → Top-k Documents**

---

## 👥 Équipe & Répartition

| Personne | Rôle | Statut |
|----------|------|--------|
| **P1 — [Ton nom]** | Data Engineer | ✅ **TERMINÉ** |
| **P2 — [Ton nom]** | Speech AI Engineer | ✅ **TERMINÉ** |
| **P3** | Machine Learning Engineer | 🔲 À faire |
| **P4** | Backend & Deployment | 🔲 À faire |

---

## 📁 Structure du projet

```
SRS-project/
│
├── data/
│   ├── audio_clean/          ← 4985 fichiers .wav (P1 ✅)
│   ├── chunks/               ← Fichiers .txt des chunks (P1 ✅)
│   ├── embeddings/           ← Matrices numpy (P2 ✅)
│   │   ├── audio_embeddings.npy
│   │   └── audio_embeddings_index.csv
│   └── output/               ← Fichiers CSV (P1 ✅)
│       ├── audio_manifest.csv
│       ├── corpus_chunks.csv
│       ├── pairs.csv
│       ├── pairs_train.csv
│       ├── pairs_val.csv
│       └── pairs_test.csv
│
├── src/
│   ├── data_pipeline.py      ← Module P1 (P1 ✅)
│   └── speech_encoder.py     ← Module P2 (P2 ✅)
│
├── notebooks/
│   ├── P1_Data_Engineer_v3.ipynb     ✅
│   ├── P2_Speech_AI_Engineer_v2.ipynb ✅
│   ├── P3_ML_Engineer.ipynb          ← À créer (P3)
│   └── P4_Backend_Deployment.ipynb   ← À créer (P4)
│
├── models/                   ← Modèles entraînés
├── api/                      ← FastAPI (P4)
├── demo/                     ← Interface Gradio (P4)
├── report/                   ← Rapport 20 pages
└── README.md
```

---

## ✅ Ce qui est déjà fait (P1 + P2)

### P1 — Dataset (TERMINÉ)

**Dataset audio** : 4985 fichiers `.wav` — 16 kHz, mono, normalisés
- Source : LibriSpeech clean-100 (3000) + clean-360 (1985)
- Durée totale : **17.61 heures**

**Corpus texte** : 6744 chunks de 512 tokens
- Source : Wikipedia (570 articles)
- Tokenizer : `sentence-transformers/all-mpnet-base-v2`

**Paires audio–document** :

| Split | Paires |
|-------|--------|
| Train | 3988 (80%) |
| Val   | 498  (10%) |
| Test  | 499  (10%) |

**Fichiers produits :**
```
data/output/audio_manifest.csv      → index de tous les audios
data/output/corpus_chunks.csv       → tous les chunks texte
data/output/pairs_train.csv         → paires d'entraînement
data/output/pairs_val.csv           → paires de validation
data/output/pairs_test.csv          → paires de test
```

---

### P2 — Speech Encoder (TERMINÉ)

**Modèle** : `facebook/wav2vec2-base-960h`  
**Output** : vecteurs normalisés L2 de dimension **768**

**Fonction principale** :
```python
from src.speech_encoder import speech_to_embedding

emb = speech_to_embedding("query.wav")
# → np.ndarray de shape (768,)
```

**SpeechEncoder class** (pour P3) :
```python
from src.speech_encoder import SpeechEncoder

speech_enc = SpeechEncoder(frozen=True).to(device)
emb = speech_enc(input_values)   # [B, 768]
```

**Charger les embeddings pré-calculés** (pour P3 et P4) :
```python
import numpy as np
import pandas as pd

# Matrice complète [4985, 768]
embeddings = np.load("data/embeddings/audio_embeddings.npy")

# Index : audio_id → row
index = pd.read_csv("data/embeddings/audio_embeddings_index.csv")

# Paires enrichies avec colonne embedding_row
pairs = pd.read_csv("data/output/pairs_train_with_embeddings.csv")

# Récupérer un vecteur audio
row = int(pairs.iloc[0]["embedding_row"])
vec = embeddings[row]   # (768,)
```

---

## 🔲 Ce qu'il reste à faire

---

### 👤 P3 — Machine Learning Engineer

**Objectif** : Entraîner le **Dual Encoder** (speech + text dans le même espace vectoriel).

#### Entrées disponibles depuis P1 + P2
```python
# 1. Embeddings audio pré-calculés
audio_embs = np.load("data/embeddings/audio_embeddings.npy")  # [4985, 768]

# 2. Corpus texte à encoder
chunks_df = pd.read_csv("data/output/corpus_chunks.csv")

# 3. Paires d'entraînement
train_df = pd.read_csv("data/output/pairs_train_with_embeddings.csv")
val_df   = pd.read_csv("data/output/pairs_val_with_embeddings.csv")
```

#### Tâches à implémenter

**1. Text Encoder**
```python
from sentence_transformers import SentenceTransformer

text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
text_emb = text_encoder.encode("Neural networks learn representations")
# → (768,)
```

**2. Dual Encoder**
```
SpeechEncoder (P2)          TextEncoder (MPNet/BGE)
      ↓                              ↓
Speech Embedding [B, 768]   Text Embedding [B, 768]
      ↓                              ↓
      └────── Cosine Similarity ─────┘
                    ↓
              Contrastive Loss
```

**3. Loss function** : InfoNCE / Contrastive Loss
```python
def contrastive_loss(speech_emb, text_emb, temperature=0.07):
    # speech_emb : [B, 768] — déjà normalisé L2
    # text_emb   : [B, 768] — déjà normalisé L2
    logits = (speech_emb @ text_emb.T) / temperature  # [B, B]
    labels = torch.arange(B).to(device)               # diagonale = positifs
    loss   = F.cross_entropy(logits, labels)
    return loss
```

**4. Hyperparamètres**
```python
epochs        = 20
batch_size    = 16
learning_rate = 2e-5
temperature   = 0.07
optimizer     = AdamW
scheduler     = linear warmup
```

**5. Métriques d'évaluation**
```
Recall@5   → sur pairs_test.csv
Recall@10  → sur pairs_test.csv
MRR        → Mean Reciprocal Rank
```

**6. Output attendu**
```
models/dual_encoder_best.pt      ← checkpoint final
models/text_embeddings.npy       ← embeddings texte de tout le corpus [N, 768]
```

---

### 👤 P4 — Backend & Deployment Engineer

**Objectif** : Construire l'index FAISS + l'API + l'interface démo.

#### Entrées disponibles depuis P1, P2, P3
```python
# Embeddings texte de tout le corpus (produits par P3)
text_embs = np.load("models/text_embeddings.npy")        # [N, 768]
chunks_df = pd.read_csv("data/output/corpus_chunks.csv") # textes associés

# Speech encoder de P2
from src.speech_encoder import speech_to_embedding
```

#### Tâches à implémenter

**1. Index FAISS**
```python
import faiss

d = 768
index = faiss.IndexFlatIP(d)          # Inner Product = cosine sur vecteurs normalisés
index.add(text_embs.astype("float32"))
faiss.write_index(index, "models/faiss_index.bin")

# Recherche
query_emb = speech_to_embedding("query.wav")              # (768,)
D, I = index.search(query_emb.reshape(1, -1), k=5)        # top-5
top_docs = chunks_df.iloc[I[0]]
```

**2. API FastAPI**
```
POST /search
    body : { audio_base64: "..." }
    return : [ { chunk_id, title, text, score }, ... ]

POST /upload_audio
    body : fichier .wav
    return : { embedding: [...] }
```

**3. Interface Gradio**
```python
import gradio as gr

def search_from_audio(audio_file):
    emb      = speech_to_embedding(audio_file)
    D, I     = index.search(emb.reshape(1,-1), k=5)
    results  = chunks_df.iloc[I[0]]
    return results[["title", "text"]].to_string()

gr.Interface(
    fn     = search_from_audio,
    inputs = gr.Audio(type="filepath"),
    outputs= gr.Textbox(label="Top 5 documents"),
    title  = "🎙️ Speech-to-Retrieval Demo"
).launch()
```

**4. Export ONNX** (optionnel, pour accélérer l'inférence)
```python
# P2 a déjà la fonction prête :
from src.speech_encoder import export_to_onnx
export_to_onnx("models/speech_encoder.onnx")
```

---

## 🗓️ Planning de synchronisation

| Jour | Sync | Personnes | Objectif |
|------|------|-----------|---------|
| Jour 1 | Setup projet | Tous | Cloner le repo, vérifier l'env |
| Jour 4 | ✅ **FAIT** | P1, P2, P3 | Dataset + embeddings validés |
| Jour 6 | ✅ **FAIT** | P1, P2, P3 | Dataset final prêt |
| Jour 7 | 🔲 | **P3 + P4** | P3 livre `text_embeddings.npy` à P4 |
| Jour 8 | 🔲 | Tous | Test pipeline complet |
| Jour 9 | 🔲 | Tous | Bugs + optimisation |
| Jour 10 | 🔲 | Tous | Démo finale + livraison |

---

## 🛠️ Installation

```bash
# Cloner le repo
git clone https://github.com/votre-org/SRS-project.git
cd SRS-project

# Créer l'environnement
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
librosa>=0.10.0
soundfile>=0.12.1
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
fastapi>=0.104.0
uvicorn>=0.24.0
gradio>=4.0.0
onnx>=1.14.0
onnxruntime>=1.16.0
tqdm>=4.65.0
joblib>=1.3.0
mlflow>=2.7.0
wikipediaapi>=0.6.0
```

---

## 📊 Architecture globale

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE (Training)                   │
│                                                         │
│  Audio .wav ──► Wav2Vec2 ──► Speech Emb [768]          │
│       (P1)         (P2)                    │            │
│                                            ▼            │
│  Text .txt  ──► MPNet    ──► Text Emb [768]            │
│       (P1)         (P3)         │          │            │
│                                 ▼          ▼            │
│                          Contrastive Loss (P3)          │
│                          Dual Encoder Training          │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE (Inference)                   │
│                                                         │
│  🎤 Microphone                                          │
│       │                                                 │
│       ▼                                                 │
│  Wav2Vec2 (P2) ──► Query Emb [768]                     │
│                          │                              │
│                          ▼                              │
│                   FAISS Index (P4) ──► Top-5 Docs      │
│                                              │          │
│                                              ▼          │
│                                      Gradio UI (P4)     │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Rapport — Répartition des sections

| Personne | Section | Pages |
|----------|---------|-------|
| P1 | Introduction + Dataset & Preprocessing | ~5 pages |
| P2 | Speech Encoder (Wav2Vec2, embeddings) | ~5 pages |
| P3 | Dual Encoder, Training, Évaluation | ~5 pages |
| P4 | Retrieval System, API, Démo, Conclusion | ~5 pages |

---

## 🎬 Vidéo démo — Script (3 min)

```
0:00 → 0:30  Présentation du problème + architecture (slides)
0:30 → 1:30  Démonstration live :
             - Parler dans le micro : "Explain neural networks"
             - Système → embedding → FAISS → Top 5 docs
1:30 → 2:30  Résultats : Recall@5, Recall@10, MRR
2:30 → 3:00  Conclusion + perspectives (Darija, multilingue)
```

---

## ❓ FAQ pour P3 et P4

**Q : Les embeddings audio sont-ils déjà calculés ?**  
R : Oui. `data/embeddings/audio_embeddings.npy` contient la matrice [4985, 768]. Pas besoin de relancer P2.

**Q : Quel text encoder utiliser ?**  
R : `sentence-transformers/all-mpnet-base-v2` — déjà utilisé par P1 pour le tokenizer, cohérent avec la dimension 768.

**Q : Les embeddings sont-ils normalisés ?**  
R : Oui, tous les vecteurs P2 sont normalisés L2. Pour FAISS utiliser `IndexFlatIP` (produit scalaire = cosine sur vecteurs normalisés).

**Q : Où mettre les modèles entraînés ?**  
R : Dans `models/` — créer le dossier s'il n'existe pas.

---

*Projet INPT — Speech-to-Retrieval System using Deep Learning*
