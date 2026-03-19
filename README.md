# Speech-to-Retrieval Project

Ce dépôt contient maintenant deux niveaux d'implémentation :

- le prototype initial de recherche sémantique texte dans `utils/`
- l'ossature de training de la personne 3 dans `training/`

## Ce qui existe déjà

Le dossier `utils/` construit des embeddings texte depuis `data/documents/ss-corpus-fr.tsv`, indexe ces embeddings avec FAISS, puis exécute une recherche top-k à partir d'une requête texte.

Ce pipeline est utile comme preuve de concept de retrieval, mais il ne couvre pas encore l'entraînement speech-to-text demandé par le projet.

## Nouveau travail ajouté pour la personne 3

Le dossier `training/` contient :

- `data.py` : chargement des text chunks, paires et embeddings audio
- `models.py` : modèle dual encoder speech/text
- `losses.py` : contrastive loss et triplet loss
- `metrics.py` : Recall@5, Recall@10, MRR
- `train_dual_encoder.py` : entraînement
- `export_text_embeddings.py` : export des embeddings texte pour la recherche
- `validate_handoff.py` : vérification du handoff de la personne 2
- `PERSON2_HANDOFF.md` : contrat d'interface pour brancher le travail de la personne 2

## Formats d'entrée attendus pour la personne 3

### `text_chunks.csv`

Colonnes obligatoires :

- `chunk_id`
- `document_id`
- `text`

Exemple :

```csv
chunk_id,document_id,text
doc_0001_chunk_00,doc_0001,"Neural networks are models inspired by the brain."
doc_0001_chunk_01,doc_0001,"They are trained with gradient-based optimization."
```

### `pairs.csv`

Colonnes minimales :

- `audio_id`
- `document_id`

Colonnes recommandées :

- `chunk_id`
- `split`

Si `chunk_id` n'est pas fourni, le pipeline prend le premier chunk du document comme positif.

Exemple :

```csv
audio_id,document_id,chunk_id,split
cv_en_0001,doc_0001,doc_0001_chunk_00,train
cv_en_0002,doc_0044,doc_0044_chunk_03,val
```

### `audio_manifest.csv`

Colonnes minimales :

- `audio_id`

Le manifest doit être aligné ligne par ligne avec `audio_embeddings.npy`.

### `audio_embeddings.npy`

- tableau numpy 2D
- shape `(N, D)`
- une ligne par `audio_id`

## Commande d'entraînement

```bash
python -m training.train_dual_encoder \
  --text-chunks-csv data/processed/text_chunks.csv \
  --pairs-csv data/processed/pairs.csv \
  --audio-manifest-csv data/processed/audio_manifest.csv \
  --audio-embeddings-npy data/processed/audio_embeddings.npy \
  --output-dir models/dual_encoder_mpnet \
  --text-model-name sentence-transformers/all-mpnet-base-v2 \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 2e-5
```

## Export des embeddings texte pour la recherche

Après entraînement :

```bash
python -m training.export_text_embeddings \
  --text-chunks-csv data/processed/text_chunks.csv \
  --checkpoint models/dual_encoder_mpnet/best_model.pt \
  --output-embeddings-npy embeddings/text_chunk_embeddings.npy \
  --output-manifest-csv embeddings/text_chunk_manifest.csv
```

## Travail attendu de la personne 2

Les instructions de handoff sont dans [training/PERSON2_HANDOFF.md](training/PERSON2_HANDOFF.md).
