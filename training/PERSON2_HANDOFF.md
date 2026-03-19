# Personne 2 -> Personne 3 Handoff

Ce document fixe l'interface de connexion entre le speech encoder et le pipeline de training.

## Ce que la personne 2 doit livrer

La personne 2 doit produire deux fichiers synchronisés :

1. `audio_manifest.csv`
2. `audio_embeddings.npy`

Le pipeline de training suppose que chaque ligne du manifest correspond exactement à une ligne du tableau numpy.

## Format de `audio_manifest.csv`

Colonnes obligatoires :

- `audio_id`

Colonnes fortement recommandées :

- `audio_path`
- `split`
- `source_model`
- `sample_rate`
- `language`

Exemple :

```csv
audio_id,audio_path,split,source_model,sample_rate,language
cv_en_0001,data/audio/train/cv_en_0001.wav,train,facebook/wav2vec2-base,16000,en
cv_en_0002,data/audio/train/cv_en_0002.wav,train,facebook/wav2vec2-base,16000,en
cv_en_1001,data/audio/val/cv_en_1001.wav,val,facebook/wav2vec2-base,16000,en
```

## Format de `audio_embeddings.npy`

- Type : `float32`
- Shape : `(N, D)`
- `N` = nombre de lignes de `audio_manifest.csv`
- `D` = dimension fixe du speech encoder

Exemple :

```python
embeddings.shape == (5000, 768)
```

## Contrat de jointure

La jointure entre le travail de la personne 2 et celui de la personne 3 se fait sur `audio_id`.

Le fichier `pairs.csv` utilisé par la personne 3 doit donc contenir la même valeur `audio_id`.

Exemple :

```csv
audio_id,document_id,chunk_id,split
cv_en_0001,doc_0042,doc_0042_chunk_00,train
cv_en_0002,doc_0187,doc_0187_chunk_03,train
cv_en_1001,doc_0200,doc_0200_chunk_01,val
```

## Fonction attendue côté personne 2

Même si la personne 3 entraîne avec des embeddings pré-calculés, la personne 2 doit aussi garder une fonction réutilisable :

```python
def speech_to_embedding(audio_file: str) -> np.ndarray:
    ...
```

Contraintes :

- entrée : fichier `.wav`
- sample rate : `16 kHz`
- sortie : vecteur 1D
- même dimension que les lignes de `audio_embeddings.npy`
- même prétraitement entre train, val, test et démo

## Checklist de validation avant handoff

- tous les audios ont été convertis en `wav`, `16 kHz`, mono
- les audios trop courts et bruités ont été retirés
- `audio_manifest.csv` et `audio_embeddings.npy` ont le même nombre de lignes
- aucune embedding ne contient `NaN`
- la dimension est fixe pour tous les audios
- les `audio_id` de `pairs.csv` existent dans `audio_manifest.csv`

## Commande de training côté personne 3

Quand les fichiers sont prêts, la personne 3 peut lancer :

```bash
python -m training.train_dual_encoder \
  --text-chunks-csv data/processed/text_chunks.csv \
  --pairs-csv data/processed/pairs.csv \
  --audio-manifest-csv data/processed/audio_manifest.csv \
  --audio-embeddings-npy data/processed/audio_embeddings.npy \
  --output-dir models/dual_encoder_mpnet
```

## Commande de validation avant intégration

Avant le training, la personne 3 peut vérifier le handoff avec :

```bash
python -m training.validate_handoff \
  --audio-manifest-csv data/processed/audio_manifest.csv \
  --audio-embeddings-npy data/processed/audio_embeddings.npy \
  --pairs-csv data/processed/pairs.csv
```

## Si la personne 2 veut brancher un nouveau modèle audio

Si la dimension change, rien ne casse côté personne 3.

Le script de training lit automatiquement la dimension réelle depuis `audio_embeddings.npy` et adapte la couche `audio_projection`.
