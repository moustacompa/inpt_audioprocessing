-- =============================================================================
-- snowflake_pipeline.sql
-- Speech-to-Retrieval (S2R) -- Pipeline complet sur Snowflake
--
-- Sections :
--   1.  Infrastructure         (warehouse, database, schemas, roles)
--   2.  Tables                 (corpus, audio, paires, embeddings)
--   3.  Stages & File Formats  (chargement des CSV, NPY, modeles)
--   4.  Chargement des donnees (COPY INTO)
--   5.  Vues utilitaires
--   6.  UDF Python Snowpark    (encodage audio, encodage texte, recherche)
--   7.  Stored Procedures      (pipeline complet, export, inference)
--   8.  Recherche vectorielle  (VECTOR_COSINE_SIMILARITY)
--   9.  Requetes d'analyse
--   10. Nettoyage (DROP optionnel)
--
-- Prerequis :
--   - Compte Snowflake avec acces Snowpark Python
--   - Fichiers CSV deposes dans un stage S3/Azure/GCS ou stage interne
--   - Modele entraine (best_model.pt) depose dans @S2R_MODELS_STAGE
--   - Python 3.10+ via Snowpark (PACKAGES requis listes en section 6)
-- =============================================================================


-- =============================================================================
-- SECTION 1 : INFRASTRUCTURE
-- =============================================================================

-- Warehouse dedie au projet (taille XS suffisante pour le dev)
CREATE WAREHOUSE IF NOT EXISTS S2R_WH
    WAREHOUSE_SIZE    = 'X-SMALL'
    AUTO_SUSPEND      = 120
    AUTO_RESUME       = TRUE
    COMMENT           = 'Warehouse Speech-to-Retrieval INPT';

USE WAREHOUSE S2R_WH;

-- Base de donnees et schemas
CREATE DATABASE IF NOT EXISTS S2R_DB
    COMMENT = 'Speech-to-Retrieval -- Projet INPT Deep Learning';

USE DATABASE S2R_DB;

CREATE SCHEMA IF NOT EXISTS S2R_DB.RAW
    COMMENT = 'Donnees brutes chargees depuis les CSV P1/P2';

CREATE SCHEMA IF NOT EXISTS S2R_DB.EMBEDDINGS
    COMMENT = 'Embeddings vectoriels (audio 768D, texte projete 256D)';

CREATE SCHEMA IF NOT EXISTS S2R_DB.TRAINING
    COMMENT = 'Paires entrainement/validation/test et metriques';

CREATE SCHEMA IF NOT EXISTS S2R_DB.INFERENCE
    COMMENT = 'Resultats de recherche et logs d inference';

CREATE SCHEMA IF NOT EXISTS S2R_DB.ML
    COMMENT = 'UDFs Snowpark et stored procedures du pipeline S2R';

-- Role applicatif
CREATE ROLE IF NOT EXISTS S2R_ROLE;
GRANT USAGE  ON WAREHOUSE S2R_WH     TO ROLE S2R_ROLE;
GRANT ALL    ON DATABASE  S2R_DB     TO ROLE S2R_ROLE;
GRANT ALL    ON ALL SCHEMAS IN DATABASE S2R_DB TO ROLE S2R_ROLE;


-- =============================================================================
-- SECTION 2 : TABLES
-- =============================================================================

USE SCHEMA S2R_DB.RAW;

-- --------------------------------------------------------------------------
-- 2.1 Corpus texte (chunks Wikipedia)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.RAW.CORPUS_CHUNKS (
    CHUNK_ID      VARCHAR(64)    NOT NULL,   -- ex : doc00000_chunk000
    DOC_IDX       INTEGER,                   -- index numerique du document
    CHUNK_IDX     INTEGER,                   -- index du chunk dans le document
    DOCUMENT_ID   VARCHAR(64)                -- derive : doc00000
                  AS (REGEXP_REPLACE(CHUNK_ID, '_chunk[0-9]+$', ''))
                  VIRTUAL,
    TITLE         VARCHAR(256),              -- titre du document Wikipedia
    SOURCE        VARCHAR(64),               -- ex : wikipedia
    TEXT          TEXT           NOT NULL,   -- contenu textuel du chunk
    TOKEN_COUNT   INTEGER,                   -- nombre de tokens
    LOADED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_CHUNKS PRIMARY KEY (CHUNK_ID)
);

-- --------------------------------------------------------------------------
-- 2.2 Manifest audio (fichiers LibriSpeech nettoyes)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.RAW.AUDIO_MANIFEST (
    AUDIO_ID      VARCHAR(64)    NOT NULL,   -- ex : libri100_01383
    FILENAME      VARCHAR(256),
    FILEPATH      VARCHAR(512),
    TRANSCRIPTION TEXT,                      -- transcription ASR de reference
    SOURCE        VARCHAR(64),               -- ex : librispeech_train.100
    DURATION_SEC  FLOAT,
    SNR_DB        FLOAT,
    LOADED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_AUDIO PRIMARY KEY (AUDIO_ID)
);

-- --------------------------------------------------------------------------
-- 2.3 Paires audio-texte (train / val / test)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.TRAINING.PAIRS (
    PAIR_ID       INTEGER AUTOINCREMENT PRIMARY KEY,
    AUDIO_FILE    VARCHAR(256),
    AUDIO_ID      VARCHAR(64)
                  AS (REGEXP_REPLACE(AUDIO_FILE, '\\.wav$', ''))
                  VIRTUAL,
    AUDIO_PATH    VARCHAR(512),
    QUERY_TEXT    TEXT,
    CHUNK_ID      VARCHAR(64),
    DOC_TITLE     VARCHAR(256),
    MATCH_SCORE   FLOAT,
    SOURCE_AUDIO  VARCHAR(64),
    SPLIT         VARCHAR(16)    DEFAULT 'train',   -- train | val | test
    LOADED_AT     TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
);

-- --------------------------------------------------------------------------
-- 2.4 Embeddings audio Wav2Vec2 (768D, pre-calcules)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS (
    AUDIO_ID          VARCHAR(64)      NOT NULL,
    EMBEDDING_INDEX   INTEGER,
    EMBEDDING_768D    VECTOR(FLOAT, 768),   -- vecteur Wav2Vec2 L2-normalise
    LOADED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_AUDIO_EMB PRIMARY KEY (AUDIO_ID)
);

-- --------------------------------------------------------------------------
-- 2.5 Embeddings texte projetes (256D, apres dual encoder)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS (
    CHUNK_ID      VARCHAR(64)      NOT NULL,
    DOCUMENT_ID   VARCHAR(64),
    TEXT_PREVIEW  VARCHAR(512),             -- 512 premiers caracteres
    EMBEDDING_256D VECTOR(FLOAT, 256),      -- espace partage dual encoder
    LOADED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_TEXT_EMB PRIMARY KEY (CHUNK_ID)
);

-- --------------------------------------------------------------------------
-- 2.6 Embeddings texte MiniLM pre-calcules (384D, avant projection)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.EMBEDDINGS.PRECOMPUTED_TEXT_EMBEDDINGS (
    CHUNK_ID        VARCHAR(64)      NOT NULL,
    EMBEDDING_INDEX INTEGER,
    EMBEDDING_384D  VECTOR(FLOAT, 384),     -- MiniLM L6-v2 brut
    LOADED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_PRE_TEXT_EMB PRIMARY KEY (CHUNK_ID)
);

-- --------------------------------------------------------------------------
-- 2.7 Resultats de recherche (log des requetes)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.INFERENCE.SEARCH_LOGS (
    LOG_ID          INTEGER AUTOINCREMENT PRIMARY KEY,
    SESSION_ID      VARCHAR(64),
    QUERY_AUDIO_ID  VARCHAR(256),
    QUERY_TEXT      TEXT,
    QUERY_MODE      VARCHAR(32),            -- audio | text
    K               INTEGER,
    RESULT_RANK     INTEGER,
    RESULT_CHUNK_ID VARCHAR(64),
    RESULT_SCORE    FLOAT,
    RESULT_TEXT     TEXT,
    SEARCHED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- --------------------------------------------------------------------------
-- 2.8 Metriques d'entrainement
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS S2R_DB.TRAINING.METRICS (
    RUN_ID          VARCHAR(64)    NOT NULL DEFAULT UUID_STRING(),
    EPOCH           INTEGER,
    TRAIN_LOSS      FLOAT,
    RECALL_AT_5     FLOAT,
    RECALL_AT_10    FLOAT,
    MRR             FLOAT,
    IS_BEST         BOOLEAN        DEFAULT FALSE,
    RECORDED_AT     TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
);


-- =============================================================================
-- SECTION 3 : STAGES ET FILE FORMATS
-- =============================================================================

USE SCHEMA S2R_DB.RAW;

-- Stage interne pour les CSV (donnees P1/P2)
CREATE STAGE IF NOT EXISTS S2R_DB.RAW.S2R_DATA_STAGE
    COMMENT = 'Fichiers CSV du projet S2R (corpus_chunks, audio_manifest, pairs)';

-- Stage pour les modeles et embeddings numpy
CREATE STAGE IF NOT EXISTS S2R_DB.ML.S2R_MODELS_STAGE
    COMMENT = 'Modeles entraines (best_model.pt) et embeddings .npy';

-- Format CSV standard
CREATE FILE FORMAT IF NOT EXISTS S2R_DB.RAW.CSV_FORMAT
    TYPE             = 'CSV'
    FIELD_DELIMITER  = ','
    RECORD_DELIMITER = '\n'
    SKIP_HEADER      = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF          = ('NULL', 'null', '')
    EMPTY_FIELD_AS_NULL = TRUE
    ENCODING         = 'UTF-8';

-- Format Parquet (pour export en masse)
CREATE FILE FORMAT IF NOT EXISTS S2R_DB.RAW.PARQUET_FORMAT
    TYPE = 'PARQUET';


-- =============================================================================
-- SECTION 4 : CHARGEMENT DES DONNEES
-- =============================================================================

-- --------------------------------------------------------------------------
-- 4.1 Charger corpus_chunks.csv
--     Avant : PUT file://data/output/corpus_chunks.csv @S2R_DB.RAW.S2R_DATA_STAGE
-- --------------------------------------------------------------------------
COPY INTO S2R_DB.RAW.CORPUS_CHUNKS (
    CHUNK_ID, DOC_IDX, CHUNK_IDX, TITLE, SOURCE, TEXT, TOKEN_COUNT
)
FROM (
    SELECT
        $1,                          -- chunk_id
        TRY_TO_NUMBER($2),           -- doc_idx
        TRY_TO_NUMBER($3),           -- chunk_idx
        $4,                          -- title
        $5,                          -- source
        $6,                          -- text
        TRY_TO_NUMBER($7)            -- token_count
    FROM @S2R_DB.RAW.S2R_DATA_STAGE/corpus_chunks.csv
)
FILE_FORMAT = (FORMAT_NAME = 'S2R_DB.RAW.CSV_FORMAT')
ON_ERROR    = 'CONTINUE';


-- --------------------------------------------------------------------------
-- 4.2 Charger audio_manifest.csv
-- --------------------------------------------------------------------------
COPY INTO S2R_DB.RAW.AUDIO_MANIFEST (
    AUDIO_ID, FILENAME, FILEPATH, TRANSCRIPTION, SOURCE, DURATION_SEC, SNR_DB
)
FROM (
    SELECT $1, $2, $3, $4, $5,
           TRY_TO_DOUBLE($6),
           TRY_TO_DOUBLE($7)
    FROM @S2R_DB.RAW.S2R_DATA_STAGE/audio_manifest.csv
)
FILE_FORMAT = (FORMAT_NAME = 'S2R_DB.RAW.CSV_FORMAT')
ON_ERROR    = 'CONTINUE';


-- --------------------------------------------------------------------------
-- 4.3 Charger les paires (train / val / test)
-- --------------------------------------------------------------------------
-- Train
COPY INTO S2R_DB.TRAINING.PAIRS (
    AUDIO_FILE, AUDIO_PATH, QUERY_TEXT, CHUNK_ID, DOC_TITLE, MATCH_SCORE, SOURCE_AUDIO, SPLIT
)
FROM (
    SELECT $1, $2, $3, $4, $5, TRY_TO_DOUBLE($6), $7, 'train'
    FROM @S2R_DB.RAW.S2R_DATA_STAGE/pairs_train.csv
)
FILE_FORMAT = (FORMAT_NAME = 'S2R_DB.RAW.CSV_FORMAT')
ON_ERROR    = 'CONTINUE';

-- Val
COPY INTO S2R_DB.TRAINING.PAIRS (
    AUDIO_FILE, AUDIO_PATH, QUERY_TEXT, CHUNK_ID, DOC_TITLE, MATCH_SCORE, SOURCE_AUDIO, SPLIT
)
FROM (
    SELECT $1, $2, $3, $4, $5, TRY_TO_DOUBLE($6), $7, 'val'
    FROM @S2R_DB.RAW.S2R_DATA_STAGE/pairs_val.csv
)
FILE_FORMAT = (FORMAT_NAME = 'S2R_DB.RAW.CSV_FORMAT')
ON_ERROR    = 'CONTINUE';

-- Test
COPY INTO S2R_DB.TRAINING.PAIRS (
    AUDIO_FILE, AUDIO_PATH, QUERY_TEXT, CHUNK_ID, DOC_TITLE, MATCH_SCORE, SOURCE_AUDIO, SPLIT
)
FROM (
    SELECT $1, $2, $3, $4, $5, TRY_TO_DOUBLE($6), $7, 'test'
    FROM @S2R_DB.RAW.S2R_DATA_STAGE/pairs_test.csv
)
FILE_FORMAT = (FORMAT_NAME = 'S2R_DB.RAW.CSV_FORMAT')
ON_ERROR    = 'CONTINUE';


-- =============================================================================
-- SECTION 5 : VUES UTILITAIRES
-- =============================================================================

USE SCHEMA S2R_DB.RAW;

-- Vue : paires enrichies avec texte du chunk et metadonnees audio
CREATE OR REPLACE VIEW S2R_DB.TRAINING.V_PAIRS_ENRICHED AS
SELECT
    p.PAIR_ID,
    p.SPLIT,
    p.AUDIO_ID,
    a.DURATION_SEC,
    a.SNR_DB,
    p.CHUNK_ID,
    c.TITLE         AS DOC_TITLE,
    c.TOKEN_COUNT,
    p.MATCH_SCORE,
    p.QUERY_TEXT,
    c.TEXT          AS CHUNK_TEXT
FROM S2R_DB.TRAINING.PAIRS     p
LEFT JOIN S2R_DB.RAW.AUDIO_MANIFEST  a ON a.AUDIO_ID = p.AUDIO_ID
LEFT JOIN S2R_DB.RAW.CORPUS_CHUNKS   c ON c.CHUNK_ID = p.CHUNK_ID;

-- Vue : statistiques du dataset
CREATE OR REPLACE VIEW S2R_DB.TRAINING.V_DATASET_STATS AS
SELECT
    SPLIT,
    COUNT(*)                          AS N_PAIRS,
    COUNT(DISTINCT AUDIO_ID)          AS N_UNIQUE_AUDIO,
    COUNT(DISTINCT CHUNK_ID)          AS N_UNIQUE_CHUNKS,
    ROUND(AVG(MATCH_SCORE), 4)        AS AVG_MATCH_SCORE,
    ROUND(MIN(MATCH_SCORE), 4)        AS MIN_MATCH_SCORE,
    ROUND(MAX(MATCH_SCORE), 4)        AS MAX_MATCH_SCORE
FROM S2R_DB.TRAINING.PAIRS
GROUP BY SPLIT
ORDER BY SPLIT;

-- Vue : top chunks les plus cites dans les paires
CREATE OR REPLACE VIEW S2R_DB.TRAINING.V_TOP_CHUNKS AS
SELECT
    p.CHUNK_ID,
    c.TITLE,
    COUNT(*)  AS CITATION_COUNT,
    ROUND(AVG(p.MATCH_SCORE), 4) AS AVG_SCORE
FROM S2R_DB.TRAINING.PAIRS     p
JOIN S2R_DB.RAW.CORPUS_CHUNKS  c ON c.CHUNK_ID = p.CHUNK_ID
GROUP BY p.CHUNK_ID, c.TITLE
ORDER BY CITATION_COUNT DESC;


-- =============================================================================
-- SECTION 6 : UDF PYTHON SNOWPARK
-- =============================================================================

USE SCHEMA S2R_DB.ML;

-- --------------------------------------------------------------------------
-- 6.1 UDF : encoder un texte en vecteur 384D (MiniLM, sans projection)
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION S2R_DB.ML.ENCODE_TEXT_MINILM(TEXT_INPUT VARCHAR)
RETURNS ARRAY
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('sentence-transformers', 'numpy')
HANDLER = 'encode'
AS $$
import numpy as np
from sentence_transformers import SentenceTransformer
_model = None

def encode(text_input: str):
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = _model.encode(text_input, normalize_embeddings=True)
    return emb.tolist()
$$;

-- --------------------------------------------------------------------------
-- 6.2 UDF : similarite cosinus entre deux vecteurs (ARRAY)
-- Utilise quand les embeddings sont stockes en ARRAY plutot qu en VECTOR
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION S2R_DB.ML.COSINE_SIMILARITY(A ARRAY, B ARRAY)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('numpy')
HANDLER = 'cosine_sim'
AS $$
import numpy as np

def cosine_sim(a, b):
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
$$;

-- --------------------------------------------------------------------------
-- 6.3 UDF vectorisee : batch encoding de textes (retourne TABLE)
-- Plus efficace que l UDF scalaire pour les grandes tables
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION S2R_DB.ML.ENCODE_TEXTS_BATCH(TEXTS ARRAY)
RETURNS TABLE (TEXT_INPUT VARCHAR, EMBEDDING ARRAY)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('sentence-transformers', 'numpy')
HANDLER = 'EncoderHandler'
AS $$
from sentence_transformers import SentenceTransformer
import numpy as np

class EncoderHandler:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def process(self, texts):
        embeddings = self.model.encode(list(texts), normalize_embeddings=True, batch_size=64)
        for text, emb in zip(texts, embeddings):
            yield (text, emb.tolist())
$$;

-- --------------------------------------------------------------------------
-- 6.4 UDF : recherche top-k dans les embeddings texte stockes
-- Retourne les k chunks les plus proches d un vecteur requete
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION S2R_DB.ML.SEARCH_TOP_K(
    QUERY_EMBEDDING VECTOR(FLOAT, 256),
    K               INTEGER
)
RETURNS TABLE (
    RANK        INTEGER,
    CHUNK_ID    VARCHAR,
    DOCUMENT_ID VARCHAR,
    SCORE       FLOAT,
    TEXT        VARCHAR
)
LANGUAGE SQL
AS $$
    SELECT
        ROW_NUMBER() OVER (ORDER BY
            VECTOR_COSINE_SIMILARITY(te.EMBEDDING_256D, QUERY_EMBEDDING) DESC
        )                                                                AS RANK,
        te.CHUNK_ID,
        te.DOCUMENT_ID,
        VECTOR_COSINE_SIMILARITY(te.EMBEDDING_256D, QUERY_EMBEDDING)    AS SCORE,
        te.TEXT_PREVIEW                                                  AS TEXT
    FROM S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS te
    QUALIFY RANK <= K
    ORDER BY SCORE DESC
$$;


-- =============================================================================
-- SECTION 7 : STORED PROCEDURES
-- =============================================================================

USE SCHEMA S2R_DB.ML;

-- --------------------------------------------------------------------------
-- 7.1 Procedure : indexer tous les chunks texte (batch encoding)
-- A executer apres chargement de corpus_chunks
-- --------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE S2R_DB.ML.SP_INDEX_CORPUS()
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python', 'sentence-transformers', 'numpy')
HANDLER = 'index_corpus'
AS $$
from snowflake.snowpark import Session
from sentence_transformers import SentenceTransformer
import numpy as np

def index_corpus(session: Session) -> str:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    chunks_df = session.table('S2R_DB.RAW.CORPUS_CHUNKS') \
                       .select('CHUNK_ID', 'DOC_IDX', 'TITLE', 'TEXT') \
                       .to_pandas()

    if chunks_df.empty:
        return 'ERROR: CORPUS_CHUNKS est vide. Chargez les donnees dabord.'

    texts      = chunks_df['TEXT'].fillna('').tolist()
    chunk_ids  = chunks_df['CHUNK_ID'].tolist()

    embeddings = model.encode(texts, normalize_embeddings=True,
                              batch_size=64, show_progress_bar=True)

    rows = []
    for cid, text, emb in zip(chunk_ids, texts, embeddings):
        doc_id = cid.rsplit('_chunk', 1)[0]
        rows.append({
            'CHUNK_ID':       cid,
            'DOCUMENT_ID':    doc_id,
            'TEXT_PREVIEW':   text[:512],
            'EMBEDDING_384D': emb.tolist(),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    sp_df = session.create_dataframe(df)
    sp_df.write.mode('overwrite').save_as_table('S2R_DB.EMBEDDINGS.PRECOMPUTED_TEXT_EMBEDDINGS')

    return f'OK: {len(rows)} chunks indexes dans PRECOMPUTED_TEXT_EMBEDDINGS'
$$;


-- --------------------------------------------------------------------------
-- 7.2 Procedure : recherche audio via embedding pre-calcule
-- Simule l inference S2R avec un embedding audio passe en parametre
-- --------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE S2R_DB.ML.SP_SEARCH_BY_AUDIO_ID(
    AUDIO_ID_PARAM  VARCHAR,
    K               INTEGER,
    SESSION_ID_PARAM VARCHAR
)
RETURNS TABLE (RANK INTEGER, CHUNK_ID VARCHAR, SCORE FLOAT, TEXT VARCHAR)
LANGUAGE SQL
AS $$
DECLARE
    query_emb VECTOR(FLOAT, 768);
BEGIN
    -- Recuperer l embedding audio pre-calcule
    SELECT EMBEDDING_768D INTO query_emb
    FROM S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS
    WHERE AUDIO_ID = AUDIO_ID_PARAM
    LIMIT 1;

    IF (query_emb IS NULL) THEN
        RAISE EXCEPTION 'Audio ID introuvable : %', AUDIO_ID_PARAM;
    END IF;

    -- Log la requete
    INSERT INTO S2R_DB.INFERENCE.SEARCH_LOGS
        (SESSION_ID, QUERY_AUDIO_ID, QUERY_MODE, K, RESULT_RANK,
         RESULT_CHUNK_ID, RESULT_SCORE, RESULT_TEXT)
    SELECT
        SESSION_ID_PARAM,
        AUDIO_ID_PARAM,
        'audio_precomputed',
        K,
        res.RANK,
        res.CHUNK_ID,
        res.SCORE,
        res.TEXT
    FROM TABLE(S2R_DB.ML.SEARCH_TOP_K(
        -- Note : necessite dim alignment 768->256 via dual encoder
        -- En prototype : utilise directement les 256 premiers dims
        CAST(query_emb AS VECTOR(FLOAT, 256)),
        K
    )) res;

    -- Retourner les resultats
    RETURN TABLE (
        SELECT RESULT_RANK, RESULT_CHUNK_ID, RESULT_SCORE, RESULT_TEXT
        FROM   S2R_DB.INFERENCE.SEARCH_LOGS
        WHERE  SESSION_ID = SESSION_ID_PARAM
          AND  QUERY_AUDIO_ID = AUDIO_ID_PARAM
        ORDER BY RESULT_RANK
    );
END;
$$;


-- --------------------------------------------------------------------------
-- 7.3 Procedure : recherche par texte (comparaison / evaluation)
-- --------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE S2R_DB.ML.SP_SEARCH_BY_TEXT(
    QUERY_TEXT_PARAM VARCHAR,
    K                INTEGER
)
RETURNS TABLE (RANK INTEGER, CHUNK_ID VARCHAR, SCORE FLOAT, TEXT VARCHAR)
LANGUAGE SQL
AS $$
BEGIN
    RETURN TABLE (
        WITH query_emb AS (
            SELECT S2R_DB.ML.ENCODE_TEXT_MINILM(:QUERY_TEXT_PARAM) AS emb_array
        ),
        scored AS (
            SELECT
                te.CHUNK_ID,
                te.TEXT_PREVIEW,
                S2R_DB.ML.COSINE_SIMILARITY(
                    :QUERY_TEXT_PARAM::ARRAY,
                    te.EMBEDDING_384D::ARRAY
                ) AS SCORE
            FROM S2R_DB.EMBEDDINGS.PRECOMPUTED_TEXT_EMBEDDINGS te
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY SCORE DESC) AS RANK,
            CHUNK_ID,
            SCORE,
            TEXT_PREVIEW AS TEXT
        FROM scored
        QUALIFY RANK <= :K
        ORDER BY SCORE DESC
    );
END;
$$;


-- --------------------------------------------------------------------------
-- 7.4 Procedure : calcul des metriques Recall@K et MRR sur le jeu de test
-- --------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE S2R_DB.ML.SP_EVALUATE(K INTEGER)
RETURNS OBJECT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'evaluate'
AS $$
from snowflake.snowpark import Session
import numpy as np
import pandas as pd

def evaluate(session: Session, k: int) -> dict:
    # Charger les paires de test avec embeddings
    test_pairs = session.sql("""
        SELECT
            p.AUDIO_ID,
            p.CHUNK_ID    AS TRUE_CHUNK_ID,
            ae.EMBEDDING_768D::ARRAY AS AUDIO_EMB
        FROM S2R_DB.TRAINING.PAIRS p
        JOIN S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS ae ON ae.AUDIO_ID = p.AUDIO_ID
        WHERE p.SPLIT = 'test'
        LIMIT 500
    """).to_pandas()

    text_embs = session.sql("""
        SELECT CHUNK_ID, EMBEDDING_256D::ARRAY AS EMB
        FROM S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS
    """).to_pandas()

    if test_pairs.empty or text_embs.empty:
        return {'error': 'Donnees manquantes. Chargez les embeddings dabord.'}

    chunk_ids  = text_embs['CHUNK_ID'].tolist()
    text_matrix = np.array([list(e) for e in text_embs['EMB']], dtype=np.float32)

    recall_at_k = []
    mrr_scores  = []

    for _, row in test_pairs.iterrows():
        query = np.array(list(row['AUDIO_EMB']), dtype=np.float32)[:text_matrix.shape[1]]
        query /= max(np.linalg.norm(query), 1e-9)

        scores  = text_matrix @ query
        top_k   = np.argsort(scores)[::-1][:k]
        top_ids = [chunk_ids[i] for i in top_k]

        hit     = row['TRUE_CHUNK_ID'] in top_ids
        recall_at_k.append(float(hit))

        rank = next((r+1 for r, cid in enumerate(top_ids) if cid == row['TRUE_CHUNK_ID']), None)
        mrr_scores.append(1.0 / rank if rank else 0.0)

    result = {
        f'Recall@{k}':   round(np.mean(recall_at_k), 4),
        'MRR':           round(np.mean(mrr_scores),  4),
        'N_TEST_PAIRS':  len(test_pairs),
    }

    # Persister les metriques
    session.sql(f"""
        INSERT INTO S2R_DB.TRAINING.METRICS (EPOCH, RECALL_AT_5, RECALL_AT_10, MRR)
        VALUES (NULL, {result.get('Recall@5', result.get(f'Recall@{k}', 0))},
                      {result.get('Recall@10', 0)},
                      {result['MRR']})
    """).collect()

    return result
$$;


-- =============================================================================
-- SECTION 8 : RECHERCHE VECTORIELLE NATIVE (VECTOR_COSINE_SIMILARITY)
-- =============================================================================

USE SCHEMA S2R_DB.EMBEDDINGS;

-- --------------------------------------------------------------------------
-- 8.1 Recherche top-5 par similarite cosinus (syntaxe native Snowflake)
-- Remplacer :query_vector par l embedding de la requete
-- --------------------------------------------------------------------------

-- Exemple avec un embedding audio pre-calcule (audio_id connu)
WITH query AS (
    SELECT EMBEDDING_768D AS q
    FROM   S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS
    WHERE  AUDIO_ID = 'libri100_01383'   -- remplacer par l audio voulu
)
SELECT
    te.CHUNK_ID,
    te.DOCUMENT_ID,
    te.TEXT_PREVIEW,
    VECTOR_COSINE_SIMILARITY(te.EMBEDDING_256D, CAST(q.q AS VECTOR(FLOAT, 256))) AS SCORE
FROM S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS te
CROSS JOIN query q
ORDER BY SCORE DESC
LIMIT 5;


-- --------------------------------------------------------------------------
-- 8.2 Evaluation batch : toutes les paires de test en une requete
-- --------------------------------------------------------------------------
WITH ranked_results AS (
    SELECT
        p.PAIR_ID,
        p.AUDIO_ID,
        p.CHUNK_ID           AS TRUE_CHUNK_ID,
        te.CHUNK_ID          AS RETRIEVED_CHUNK_ID,
        VECTOR_COSINE_SIMILARITY(
            te.EMBEDDING_256D,
            CAST(ae.EMBEDDING_768D AS VECTOR(FLOAT, 256))
        )                    AS SCORE,
        ROW_NUMBER() OVER (
            PARTITION BY p.PAIR_ID
            ORDER BY VECTOR_COSINE_SIMILARITY(
                te.EMBEDDING_256D,
                CAST(ae.EMBEDDING_768D AS VECTOR(FLOAT, 256))
            ) DESC
        )                    AS RANK
    FROM S2R_DB.TRAINING.PAIRS                p
    JOIN S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS   ae ON ae.AUDIO_ID = p.AUDIO_ID
    CROSS JOIN S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS te
    WHERE p.SPLIT = 'test'
)
SELECT
    COUNT_IF(RANK <= 5  AND TRUE_CHUNK_ID = RETRIEVED_CHUNK_ID)
        / COUNT(DISTINCT PAIR_ID)                              AS "Recall@5",
    COUNT_IF(RANK <= 10 AND TRUE_CHUNK_ID = RETRIEVED_CHUNK_ID)
        / COUNT(DISTINCT PAIR_ID)                             AS "Recall@10",
    AVG(IFF(TRUE_CHUNK_ID = RETRIEVED_CHUNK_ID, 1.0 / RANK, 0)) AS "MRR"
FROM ranked_results
WHERE RANK <= 10;


-- =============================================================================
-- SECTION 9 : REQUETES D ANALYSE
-- =============================================================================

-- Statistiques generales du dataset
SELECT * FROM S2R_DB.TRAINING.V_DATASET_STATS;

-- Distribution de la duree des audios
SELECT
    FLOOR(DURATION_SEC / 5) * 5 AS DURATION_BUCKET_SEC,
    COUNT(*)                    AS N_FILES,
    ROUND(AVG(SNR_DB), 2)       AS AVG_SNR
FROM S2R_DB.RAW.AUDIO_MANIFEST
GROUP BY 1
ORDER BY 1;

-- Top 10 documents les plus references
SELECT * FROM S2R_DB.TRAINING.V_TOP_CHUNKS LIMIT 10;

-- Chunks sans embedding (a encoder)
SELECT c.CHUNK_ID, c.TITLE
FROM S2R_DB.RAW.CORPUS_CHUNKS c
LEFT JOIN S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS te ON te.CHUNK_ID = c.CHUNK_ID
WHERE te.CHUNK_ID IS NULL
LIMIT 20;

-- Derniers logs de recherche
SELECT
    SEARCHED_AT,
    QUERY_AUDIO_ID,
    QUERY_MODE,
    K,
    RESULT_RANK,
    ROUND(RESULT_SCORE, 4) AS SCORE,
    LEFT(RESULT_TEXT, 100) AS TEXT_PREVIEW
FROM S2R_DB.INFERENCE.SEARCH_LOGS
ORDER BY SEARCHED_AT DESC
LIMIT 50;

-- Evolution des metriques d entrainement
SELECT
    RECORDED_AT,
    EPOCH,
    ROUND(TRAIN_LOSS,  4) AS LOSS,
    ROUND(RECALL_AT_5, 4) AS "R@5",
    ROUND(MRR,         4) AS MRR,
    IS_BEST
FROM S2R_DB.TRAINING.METRICS
ORDER BY RECORDED_AT;

-- Couverture des embeddings
SELECT
    (SELECT COUNT(*) FROM S2R_DB.RAW.CORPUS_CHUNKS)            AS TOTAL_CHUNKS,
    (SELECT COUNT(*) FROM S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS)   AS INDEXED_CHUNKS,
    (SELECT COUNT(*) FROM S2R_DB.RAW.AUDIO_MANIFEST)           AS TOTAL_AUDIO,
    (SELECT COUNT(*) FROM S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS)  AS AUDIO_WITH_EMBEDDINGS;


-- =============================================================================
-- SECTION 10 : NETTOYAGE (a executer uniquement si necessaire)
-- =============================================================================

-- DROP TABLE S2R_DB.EMBEDDINGS.TEXT_EMBEDDINGS;
-- DROP TABLE S2R_DB.EMBEDDINGS.AUDIO_EMBEDDINGS;
-- DROP TABLE S2R_DB.EMBEDDINGS.PRECOMPUTED_TEXT_EMBEDDINGS;
-- DROP TABLE S2R_DB.TRAINING.PAIRS;
-- DROP TABLE S2R_DB.TRAINING.METRICS;
-- DROP TABLE S2R_DB.INFERENCE.SEARCH_LOGS;
-- DROP TABLE S2R_DB.RAW.CORPUS_CHUNKS;
-- DROP TABLE S2R_DB.RAW.AUDIO_MANIFEST;
-- DROP SCHEMA S2R_DB.RAW;
-- DROP SCHEMA S2R_DB.EMBEDDINGS;
-- DROP SCHEMA S2R_DB.TRAINING;
-- DROP SCHEMA S2R_DB.INFERENCE;
-- DROP SCHEMA S2R_DB.ML;
-- DROP DATABASE S2R_DB;
-- DROP WAREHOUSE S2R_WH;
