@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: run_pipeline.bat -- Pipeline complet Speech-to-Retrieval (S2R) -- Windows
::
:: Usage :
::   run_pipeline.bat               pipeline complet
::   run_pipeline.bat --skip-train  sauter l'entrainement
::   run_pipeline.bat --api         lancer l'API FastAPI a la fin
::   run_pipeline.bat --demo        lancer l'interface Gradio a la fin
::
:: Prerequis :
::   pip install -r requirements.txt
:: =============================================================================

:: ---------------------------------------------------------------------------
:: Parametres par defaut (modifiables ici)
:: ---------------------------------------------------------------------------
set TEXT_CHUNKS_CSV=data\output\corpus_chunks.csv
set PAIRS_TRAIN_CSV=data\output\pairs_train.csv
set PAIRS_VAL_CSV=data\output\pairs_val.csv
set AUDIO_MANIFEST_CSV=embeddings\audio_embeddings_index.csv
set AUDIO_EMBEDDINGS_NPY=embeddings\audio_embeddings.npy
set AUDIO_QUERIES_DIR=data\audio_queries

set OUTPUT_DIR=models\dual_encoder_mpnet
set CHECKPOINT=%OUTPUT_DIR%\best_model.pt

set TEXT_CHUNK_EMBEDDINGS=embeddings\text_chunk_embeddings.npy
set TEXT_CHUNK_MANIFEST=embeddings\text_chunk_manifest.csv

set PRECOMPUTED_TEXT_NPY=embeddings\precomputed_text.npy
set PRECOMPUTED_TEXT_MANIFEST=embeddings\precomputed_text_manifest.csv

set TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2
set PROJECTION_DIM=256
set EPOCHS=20
set BATCH_SIZE=16
set LEARNING_RATE=2e-5
set WARMUP_STEPS=100
set MAX_GRAD_NORM=1.0
set LOSS=contrastive
set TEMPERATURE=0.07
set K=5

set API_PORT=8000
set DEMO_PORT=7860

:: ---------------------------------------------------------------------------
:: Analyse des arguments
:: ---------------------------------------------------------------------------
set SKIP_TRAIN=0
set LAUNCH_API=0
set LAUNCH_DEMO=0

:parse_args
if "%~1"=="" goto end_args
if /i "%~1"=="--skip-train" set SKIP_TRAIN=1
if /i "%~1"=="--api"        set LAUNCH_API=1
if /i "%~1"=="--demo"       set LAUNCH_DEMO=1
shift
goto parse_args
:end_args

:: ---------------------------------------------------------------------------
:: Verifications preliminaires
:: ---------------------------------------------------------------------------
echo.
echo [INFO]  Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python introuvable. Activez votre virtualenv.
    exit /b 1
)
python --version

echo [INFO]  Verification des dependances...
python -c "import torch, transformers, faiss, gradio, fastapi" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Dependances manquantes. Executez : pip install -r requirements.txt
    exit /b 1
)
echo [OK]    Dependances OK

:: ---------------------------------------------------------------------------
:: ETAPE 1 -- Validation des donnees P1/P2
:: ---------------------------------------------------------------------------
echo.
echo ======================================================================
echo [INFO]  ETAPE 1 -- Validation des donnees P1/P2
echo ======================================================================

if not exist "%TEXT_CHUNKS_CSV%" (
    echo [ERROR] Fichier introuvable : %TEXT_CHUNKS_CSV%
    echo         Executez d'abord les notebooks P1.
    exit /b 1
)
if not exist "%AUDIO_EMBEDDINGS_NPY%" (
    echo [ERROR] Fichier introuvable : %AUDIO_EMBEDDINGS_NPY%
    echo         Executez d'abord le notebook P2.
    exit /b 1
)

python -m training.validate_handoff ^
    --audio-manifest-csv "%AUDIO_MANIFEST_CSV%" ^
    --audio-embeddings-npy "%AUDIO_EMBEDDINGS_NPY%" ^
    --pairs-csv "%PAIRS_TRAIN_CSV%"
if errorlevel 1 (
    echo [ERROR] Validation P1/P2 echouee.
    exit /b 1
)
echo [OK]    Donnees P1/P2 validees

:: ---------------------------------------------------------------------------
:: ETAPE 2 -- Entrainement du Dual Encoder
:: ---------------------------------------------------------------------------
echo.
echo ======================================================================
echo [INFO]  ETAPE 2 -- Entrainement du Dual Encoder
echo ======================================================================

if "%SKIP_TRAIN%"=="1" (
    echo [WARN]  --skip-train active : entrainement ignore
    if not exist "%CHECKPOINT%" (
        echo [ERROR] Checkpoint introuvable : %CHECKPOINT%
        exit /b 1
    )
    echo [OK]    Checkpoint existant : %CHECKPOINT%
) else (
    if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

    :: --- Pre-calcul des embeddings texte (une seule fois, mode rapide) ---
    echo.
    echo [INFO]  Pre-calcul des embeddings texte...
    python -m training.precompute_text ^
        --text-chunks-csv "%TEXT_CHUNKS_CSV%" ^
        --model-name "%TEXT_MODEL%" ^
        --output-npy "%PRECOMPUTED_TEXT_NPY%" ^
        --output-manifest "%PRECOMPUTED_TEXT_MANIFEST%" ^
        --batch-size 64
    if errorlevel 1 (
        echo [ERROR] Pre-calcul des embeddings texte echoue.
        exit /b 1
    )
    echo [OK]    Embeddings texte pre-calcules : %PRECOMPUTED_TEXT_NPY%

    :: --- Entrainement (mode rapide : projection uniquement) ---
    python -m training.train_dual_encoder ^
        --precomputed-text-npy "%PRECOMPUTED_TEXT_NPY%" ^
        --precomputed-text-manifest "%PRECOMPUTED_TEXT_MANIFEST%" ^
        --pairs-csv "%PAIRS_TRAIN_CSV%" ^
        --val-pairs-csv "%PAIRS_VAL_CSV%" ^
        --audio-manifest-csv "%AUDIO_MANIFEST_CSV%" ^
        --audio-embeddings-npy "%AUDIO_EMBEDDINGS_NPY%" ^
        --output-dir "%OUTPUT_DIR%" ^
        --projection-dim %PROJECTION_DIM% ^
        --epochs %EPOCHS% ^
        --batch-size %BATCH_SIZE% ^
        --learning-rate %LEARNING_RATE% ^
        --warmup-steps %WARMUP_STEPS% ^
        --max-grad-norm %MAX_GRAD_NORM% ^
        --loss %LOSS% ^
        --temperature %TEMPERATURE% ^
        --num-workers 0
    if errorlevel 1 (
        echo [ERROR] Entrainement echoue.
        exit /b 1
    )
    echo [OK]    Entrainement termine : %CHECKPOINT%
)

:: ---------------------------------------------------------------------------
:: ETAPE 3 -- Export des embeddings texte
:: ---------------------------------------------------------------------------
echo.
echo ======================================================================
echo [INFO]  ETAPE 3 -- Export des embeddings texte
echo ======================================================================

python -m training.export_text_embeddings ^
    --precomputed-text-npy "%PRECOMPUTED_TEXT_NPY%" ^
    --precomputed-text-manifest "%PRECOMPUTED_TEXT_MANIFEST%" ^
    --text-chunks-csv "%TEXT_CHUNKS_CSV%" ^
    --checkpoint "%CHECKPOINT%" ^
    --output-embeddings-npy "%TEXT_CHUNK_EMBEDDINGS%" ^
    --output-manifest-csv "%TEXT_CHUNK_MANIFEST%" ^
    --batch-size 64
if errorlevel 1 (
    echo [ERROR] Export des embeddings texte echoue.
    exit /b 1
)
echo [OK]    Embeddings texte exportes : %TEXT_CHUNK_EMBEDDINGS%

:: ---------------------------------------------------------------------------
:: ETAPE 4 -- Test d'inference
:: ---------------------------------------------------------------------------
echo.
echo ======================================================================
echo [INFO]  ETAPE 4 -- Test d'inference (top-%K% documents)
echo ======================================================================

set SAMPLE_WAV=
for %%f in ("%AUDIO_QUERIES_DIR%\*.wav") do (
    if "!SAMPLE_WAV!"=="" set SAMPLE_WAV=%%f
)

if "!SAMPLE_WAV!"=="" (
    echo [WARN]  Aucun fichier .wav trouve dans %AUDIO_QUERIES_DIR% -- etape ignoree
) else (
    echo [INFO]  Fichier de test : !SAMPLE_WAV!
    python inference.py ^
        --audio "!SAMPLE_WAV!" ^
        --checkpoint "%CHECKPOINT%" ^
        --text-embeddings "%TEXT_CHUNK_EMBEDDINGS%" ^
        --manifest "%TEXT_CHUNK_MANIFEST%" ^
        --k %K%
    if errorlevel 1 (
        echo [WARN]  Inference echouee (non bloquant)
    ) else (
        echo [OK]    Inference terminee
    )
)

:: ---------------------------------------------------------------------------
:: ETAPE 5 -- Tests du pipeline
:: ---------------------------------------------------------------------------
echo.
echo ======================================================================
echo [INFO]  ETAPE 5 -- Tests du pipeline de recherche
echo ======================================================================

python utils\test.py --mode audio ^
    --embeddings-path "%TEXT_CHUNK_EMBEDDINGS%" ^
    --manifest-path "%TEXT_CHUNK_MANIFEST%" ^
    --audio "!SAMPLE_WAV!"
if errorlevel 1 (
    echo [WARN]  Test audio echoue (non bloquant)
) else (
    echo [OK]    Test audio OK
)

python utils\test.py --mode text ^
    --query "customer support audio speech retrieval" ^
    --embeddings-path "%TEXT_CHUNK_EMBEDDINGS%" ^
    --manifest-path "%TEXT_CHUNK_MANIFEST%"
if errorlevel 1 (
    echo [WARN]  Test texte echoue (non bloquant)
) else (
    echo [OK]    Test texte OK
)

:: ---------------------------------------------------------------------------
:: ETAPE 6 -- Lancement API ou demo (optionnel)
:: ---------------------------------------------------------------------------
if "%LAUNCH_API%"=="1" (
    echo.
    echo ======================================================================
    echo [INFO]  ETAPE 6 -- Lancement de l'API FastAPI (port %API_PORT%)
    echo ======================================================================
    echo [INFO]  API       : http://localhost:%API_PORT%
    echo [INFO]  Swagger   : http://localhost:%API_PORT%/docs
    echo [INFO]  Arret     : Ctrl+C
    set S2R_CHECKPOINT=%CHECKPOINT%
    set S2R_TEXT_EMBEDDINGS=%TEXT_CHUNK_EMBEDDINGS%
    set S2R_MANIFEST=%TEXT_CHUNK_MANIFEST%
    uvicorn api.main:app --host 0.0.0.0 --port %API_PORT%
    goto end
)

if "%LAUNCH_DEMO%"=="1" (
    echo.
    echo ======================================================================
    echo [INFO]  ETAPE 6 -- Lancement de l'interface Gradio (port %DEMO_PORT%)
    echo ======================================================================
    echo [INFO]  Interface : http://localhost:%DEMO_PORT%
    echo [INFO]  Arret     : Ctrl+C
    python demo\app.py --checkpoint "%CHECKPOINT%" --port %DEMO_PORT%
    goto end
)

:: ---------------------------------------------------------------------------
:: Resume final
:: ---------------------------------------------------------------------------
echo.
echo ======================================================================
echo   Pipeline S2R termine avec succes.
echo ======================================================================
echo.
echo   Checkpoint  : %CHECKPOINT%
echo   Embeddings  : %TEXT_CHUNK_EMBEDDINGS%
echo   Manifest    : %TEXT_CHUNK_MANIFEST%
echo.
echo   Lancer la demo  : run_pipeline.bat --skip-train --demo
echo   Lancer l'API    : run_pipeline.bat --skip-train --api
echo.

:end
endlocal
exit /b 0
