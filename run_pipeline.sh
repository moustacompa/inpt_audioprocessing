#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Pipeline complet Speech-to-Retrieval (S2R)
#
# Usage :
#   bash run_pipeline.sh              # pipeline complet
#   bash run_pipeline.sh --skip-train # sauter l'entraînement (démo uniquement)
#   bash run_pipeline.sh --api        # lancer l'API FastAPI à la fin
#   bash run_pipeline.sh --demo       # lancer l'interface Gradio à la fin
#
# Prérequis :
#   pip install -r requirements.txt
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Couleurs pour les logs
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Paramètres par défaut (modifiables ici)
# ---------------------------------------------------------------------------
TEXT_CHUNKS_CSV="data/output/corpus_chunks.csv"
PAIRS_TRAIN_CSV="data/output/pairs_train.csv"
PAIRS_VAL_CSV="data/output/pairs_val.csv"
PAIRS_TEST_CSV="data/output/pairs_test.csv"
AUDIO_MANIFEST_CSV="embeddings/audio_embeddings_index.csv"
AUDIO_EMBEDDINGS_NPY="embeddings/audio_embeddings.npy"
AUDIO_CLEAN_DIR="data/audio_clean"
AUDIO_QUERIES_DIR="data/audio_queries"

OUTPUT_DIR="models/dual_encoder_mpnet"
CHECKPOINT="$OUTPUT_DIR/best_model.pt"

TEXT_CHUNK_EMBEDDINGS="embeddings/text_chunk_embeddings.npy"
TEXT_CHUNK_MANIFEST="embeddings/text_chunk_manifest.csv"

TEXT_MODEL="sentence-transformers/all-mpnet-base-v2"
PROJECTION_DIM=768
EPOCHS=20
BATCH_SIZE=16
#EPOCHS=1
#BATCH_SIZE=32
LEARNING_RATE=2e-5
WARMUP_STEPS=100
MAX_GRAD_NORM=1.0
LOSS="contrastive"
TEMPERATURE=0.07
K=5

API_PORT=8000
DEMO_PORT=7860

# ---------------------------------------------------------------------------
# Analyse des arguments
# ---------------------------------------------------------------------------
SKIP_TRAIN=false
LAUNCH_API=false
LAUNCH_DEMO=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --api)        LAUNCH_API=true ;;
        --demo)       LAUNCH_DEMO=true ;;
        *) warn "Argument inconnu : $arg (ignoré)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Vérifications préliminaires
# ---------------------------------------------------------------------------
log "Vérification de l'environnement Python..."
python --version || fail "Python introuvable. Activez votre virtualenv."

log "Vérification des dépendances..."
python -m venv .venv
source .venv/Scripts/activate
python -c "import torch, transformers, faiss, gradio, fastapi" 2>/dev/null \
    || fail "Dépendances manquantes. Exécutez : pip install -r requirements.txt"
ok "Dépendances OK"

# ---------------------------------------------------------------------------
# ÉTAPE 1 — Validation du handoff P1/P2
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
log "ÉTAPE 1 — Validation des données P1/P2"
echo "======================================================================"

if [[ ! -f "$TEXT_CHUNKS_CSV" ]]; then
    fail "Fichier introuvable : $TEXT_CHUNKS_CSV — exécutez d'abord les notebooks P1."
fi
if [[ ! -f "$AUDIO_EMBEDDINGS_NPY" ]]; then
    fail "Fichier introuvable : $AUDIO_EMBEDDINGS_NPY — exécutez d'abord le notebook P2."
fi

python -m training.validate_handoff \
    --audio-manifest-csv  "$AUDIO_MANIFEST_CSV" \
    --audio-embeddings-npy "$AUDIO_EMBEDDINGS_NPY" \
    --pairs-csv            "$PAIRS_TRAIN_CSV"

ok "Données P1/P2 validées"

# ---------------------------------------------------------------------------
# ÉTAPE 2 — Entraînement du Dual Encoder
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
log "ÉTAPE 2 — Entraînement du Dual Encoder"
echo "======================================================================"

if $SKIP_TRAIN; then
    warn "--skip-train activé : entraînement ignoré"
    if [[ ! -f "$CHECKPOINT" ]]; then
        fail "Checkpoint introuvable ($CHECKPOINT) et --skip-train activé. Entraînez d'abord le modèle."
    fi
    ok "Checkpoint existant utilisé : $CHECKPOINT"
else
    mkdir -p "$OUTPUT_DIR"

    python -m training.train_dual_encoder \
        --text-chunks-csv      "$TEXT_CHUNKS_CSV" \
        --pairs-csv            "$PAIRS_TRAIN_CSV" \
        --val-pairs-csv        "$PAIRS_VAL_CSV" \
        --audio-manifest-csv   "$AUDIO_MANIFEST_CSV" \
        --audio-embeddings-npy "$AUDIO_EMBEDDINGS_NPY" \
        --output-dir           "$OUTPUT_DIR" \
        --text-model-name      "$TEXT_MODEL" \
        --projection-dim       "$PROJECTION_DIM" \
        --epochs               "$EPOCHS" \
        --batch-size           "$BATCH_SIZE" \
        --learning-rate        "$LEARNING_RATE" \
        --warmup-steps         "$WARMUP_STEPS" \
        --max-grad-norm        "$MAX_GRAD_NORM" \
        --loss                 "$LOSS" \
        --temperature          "$TEMPERATURE"

    ok "Entraînement terminé → $CHECKPOINT"
fi

# ---------------------------------------------------------------------------
# ÉTAPE 3 — Export des embeddings texte
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
log "ÉTAPE 3 — Export des embeddings texte (espace partagé)"
echo "======================================================================"

python -m training.export_text_embeddings \
    --text-chunks-csv       "$TEXT_CHUNKS_CSV" \
    --checkpoint            "$CHECKPOINT" \
    --output-embeddings-npy "$TEXT_CHUNK_EMBEDDINGS" \
    --output-manifest-csv   "$TEXT_CHUNK_MANIFEST" \
    --batch-size            32

ok "Embeddings texte exportés → $TEXT_CHUNK_EMBEDDINGS"

# ---------------------------------------------------------------------------
# ÉTAPE 4 — Test d'inférence sur un fichier audio de référence
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
log "ÉTAPE 4 — Test d'inférence (top-$K documents)"
echo "======================================================================"

# Sélectionne le premier .wav disponible dans audio_queries/
SAMPLE_WAV=$(find "$AUDIO_QUERIES_DIR" -name "*.wav" | head -n 1)

if [[ -z "$SAMPLE_WAV" ]]; then
    warn "Aucun fichier .wav trouvé dans $AUDIO_QUERIES_DIR — étape ignorée"
else
    log "Fichier de test : $SAMPLE_WAV"
    python inference.py \
        --audio           "$SAMPLE_WAV" \
        --checkpoint      "$CHECKPOINT" \
        --text-embeddings "$TEXT_CHUNK_EMBEDDINGS" \
        --manifest        "$TEXT_CHUNK_MANIFEST" \
        --k               "$K"
    ok "Inférence terminée"
fi

# ---------------------------------------------------------------------------
# ÉTAPE 5 — Tests unitaires du pipeline de recherche
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
log "ÉTAPE 5 — Tests du pipeline de recherche"
echo "======================================================================"

python utils/test.py --mode audio \
    --embeddings-path "$TEXT_CHUNK_EMBEDDINGS" \
    --manifest-path   "$TEXT_CHUNK_MANIFEST" \
    --audio           "$SAMPLE_WAV" \
    && ok "Test audio OK" \
    || warn "Test audio échoué (non bloquant)"

python utils/test.py --mode text \
    --query "customer support audio speech retrieval" \
    --embeddings-path "$TEXT_CHUNK_EMBEDDINGS" \
    --manifest-path   "$TEXT_CHUNK_MANIFEST" \
    && ok "Test texte OK" \
    || warn "Test texte échoué (non bloquant)"

# ---------------------------------------------------------------------------
# ÉTAPE 6 — Lancement API ou démo (optionnel)
# ---------------------------------------------------------------------------
if $LAUNCH_API; then
    echo ""
    echo "======================================================================"
    log "ÉTAPE 6 — Lancement de l'API FastAPI (port $API_PORT)"
    echo "======================================================================"
    log "API disponible sur http://localhost:$API_PORT"
    log "Documentation : http://localhost:$API_PORT/docs"
    log "Arrêt : Ctrl+C"
    export S2R_CHECKPOINT="$CHECKPOINT"
    export S2R_TEXT_EMBEDDINGS="$TEXT_CHUNK_EMBEDDINGS"
    export S2R_MANIFEST="$TEXT_CHUNK_MANIFEST"
    uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT"

elif $LAUNCH_DEMO; then
    echo ""
    echo "======================================================================"
    log "ÉTAPE 6 — Lancement de l'interface Gradio (port $DEMO_PORT)"
    echo "======================================================================"
    log "Interface disponible sur http://localhost:$DEMO_PORT"
    log "Arrêt : Ctrl+C"
    python demo/app.py \
        --checkpoint "$CHECKPOINT" \
        --port "$DEMO_PORT"
fi

# ---------------------------------------------------------------------------
# Résumé final
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo -e "${GREEN}Pipeline S2R terminé avec succès.${NC}"
echo "======================================================================"
echo ""
echo "  Checkpoint  : $CHECKPOINT"
echo "  Embeddings  : $TEXT_CHUNK_EMBEDDINGS"
echo "  Manifest    : $TEXT_CHUNK_MANIFEST"
echo ""
echo "  Lancer la démo  : bash run_pipeline.sh --skip-train --demo"
echo "  Lancer l'API    : bash run_pipeline.sh --skip-train --api"
echo ""
