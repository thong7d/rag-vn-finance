# %% [markdown]
# # Phase 3: Embedding & FAISS Indexing
# 
# **Pipeline**: Vietnamese Financial News RAG System — v3  
# **Hardware**: ⚡ Google Colab T4 GPU (REQUIRED)  
# **Model**: `intfloat/multilingual-e5-large` (560M params · dim=1024)
# 
# **Parallelism**: Run this notebook on 3 separate Colab accounts simultaneously.  
# Change `TARGET_STRATEGY` below to match the account assignment:
# 
# | Account | TARGET_STRATEGY |
# |---------|-----------------|
# | Account 1 | `fixed_size` |
# | Account 2 | `sentence_aware` |
# | Account 3 | `article_level` |
# 
# All cells are **idempotent** — safe to re-run after a Colab timeout.  
# On resume, the checkpoint is detected automatically and encoding continues from where it stopped.

# %% [markdown]
# ## Cell 0 — Environment Setup & GPU Verification

# %%
import os, sys
from pathlib import Path

# ── Strategy selector (change this per Colab account) ──────────────────────
TARGET_STRATEGY = 'fixed_size'   # options: 'fixed_size' | 'sentence_aware' | 'article_level'

print(f"Target strategy: {TARGET_STRATEGY}")

# ── Environment detection ───────────────────────────────────────────────────
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()
print(f"Runtime: {'Google Colab' if IN_COLAB else 'Local'}")

# ── GPU check ──────────────────────────────────────────────────────────────
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cpu':
    print("⚠️  WARNING: No GPU detected. Embedding will be extremely slow.")
    print("   Go to Runtime → Change runtime type → GPU (T4) before continuing.")
else:
    print(f"✅ GPU ready: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Mount Drive (Colab only) ────────────────────────────────────────────────
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = Path('/content/rag-vn-finance/implementation')
    if not REPO_ROOT.exists():
        os.system('git clone https://github.com/thong7d/rag-vn-finance /content/rag-vn-finance')
    # Install dependencies
    req_path = REPO_ROOT / 'requirements.txt'
    if req_path.exists():
        os.system(f'pip install -r {req_path} -q')
        # Colab ships faiss-cpu; upgrade to faiss-gpu if on T4
        if device == 'cuda':
            os.system('pip install faiss-gpu -q')
else:
    REPO_ROOT = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())

print(f"Project root: {REPO_ROOT}")
assert REPO_ROOT.exists(), f"Project root not found: {REPO_ROOT}"

src_path = str(REPO_ROOT)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("\nCell 0 complete.")

# %% [markdown]
# ## Cell 1 — Load Config & Resolve Paths

# %%
import json
import pandas as pd
from src.utils import load_config, resolve_path

config = load_config(REPO_ROOT / 'configs' / 'config.yaml')
emb_cfg = config['embedding']

# ── Resolve paths ───────────────────────────────────────────────────────────
chunks_base = resolve_path(config['chunking'], 'output_dir')
if not os.path.isabs(chunks_base):
    chunks_base = str(REPO_ROOT / chunks_base)

emb_base = resolve_path(emb_cfg, 'output_dir')
if not os.path.isabs(emb_base):
    emb_base = str(REPO_ROOT / emb_base)

idx_base = resolve_path(config['indexing'], 'output_dir')
if not os.path.isabs(idx_base):
    idx_base = str(REPO_ROOT / idx_base)

# Per-strategy paths
CHUNKS_PATH     = os.path.join(chunks_base, TARGET_STRATEGY, 'chunks.parquet')
EMB_DIR         = os.path.join(emb_base,    TARGET_STRATEGY)
EMB_NPY_PATH    = os.path.join(EMB_DIR,     'embeddings.npy')
CHECKPOINT_PATH = os.path.join(EMB_DIR,     'checkpoint.json')
IDX_DIR         = os.path.join(idx_base,    TARGET_STRATEGY)
FAISS_PATH      = os.path.join(IDX_DIR,     'index.faiss')

print(f"Strategy         : {TARGET_STRATEGY}")
print(f"Chunks input     : {CHUNKS_PATH}")
print(f"Embeddings dir   : {EMB_DIR}")
print(f"Embeddings .npy  : {EMB_NPY_PATH}")
print(f"Checkpoint       : {CHECKPOINT_PATH}")
print(f"FAISS index      : {FAISS_PATH}")

assert os.path.exists(CHUNKS_PATH), f"❌ Chunks file not found: {CHUNKS_PATH}"
print("\n✅ Paths resolved. Cell 1 complete.")

# %% [markdown]
# ## Cell 2 — Load Chunk Dataset

# %%
df_chunks = pd.read_parquet(CHUNKS_PATH)
print(f"Chunks loaded: {len(df_chunks):,} rows")
print(f"Columns      : {df_chunks.columns.tolist()}")
print(f"Sample chunk_id: {df_chunks['chunk_id'].iloc[0]}")

# Verify required columns
required = ['chunk_id', 'doc_id', 'text', 'strategy',
            'source', 'category', 'year', 'title', 'url',
            'tickers', 'is_historical', 'numerical_density', 'entities']
missing = [c for c in required if c not in df_chunks.columns]
if missing:
    raise ValueError(f"Missing columns in chunks: {missing}")

print(f"\n✅ Schema verified. {len(df_chunks):,} chunks ready.")
df_chunks.head(2)

# %% [markdown]
# ## Cell 3 — Apply "passage: " Prefix
# 
# > **Required by intfloat/multilingual-e5-large**  
# > Documents must be prefixed with `"passage: "` and queries with `"query: "`.  
# > Missing this prefix significantly degrades retrieval quality.

# %%
from src.embedding import PASSAGE_PREFIX

# Apply prefix to every chunk text
prefixed_texts = [PASSAGE_PREFIX + str(t) for t in df_chunks['text'].tolist()]

print(f"Total texts to embed : {len(prefixed_texts):,}")
print(f"Prefix applied       : '{PASSAGE_PREFIX}'")
print(f"Sample (first 80 chars): {prefixed_texts[0][:80]}...")

print("\nCell 3 complete.")

# %% [markdown]
# ## Cell 4 — Load Embedding Model

# %%
from sentence_transformers import SentenceTransformer

model_name = emb_cfg['model_name']
print(f"Loading model: {model_name}")
print("(First run downloads ~2.2 GB — subsequent runs load from cache)")

embedding_model = SentenceTransformer(model_name, device=device)
embedding_model.max_seq_length = 512

print(f"\n✅ Model loaded on {device}")
print(f"   Embedding dimension : {embedding_model.get_sentence_embedding_dimension()}")
print(f"   Max sequence length : {embedding_model.max_seq_length}")

# %% [markdown]
# ## Cell 5 — Encode Chunks (with Checkpointing)
# 
# > ⏱️ **TIME WARNING**: ~2–3 hours on T4 for fixed_size (~42K chunks).  
# > If the session times out, just re-run this cell — it resumes from the checkpoint automatically.

# %%
from src.embedding import encode_chunks_with_checkpoint

batch_size       = emb_cfg.get('batch_size', 64)
checkpoint_every = emb_cfg.get('checkpoint_every', 100)

print(f"Batch size      : {batch_size}")
print(f"Checkpoint every: {checkpoint_every} batches")
print(f"Total chunks    : {len(prefixed_texts):,}")
print(f"Total batches   : {(len(prefixed_texts) + batch_size - 1) // batch_size:,}")

if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        ckpt = json.load(f)
    print(f"\n🔄 Resuming — {ckpt.get('completed_chunks', 0)}/{len(prefixed_texts)} chunks done")
else:
    print("\n🚀 Starting fresh encoding run...")

embeddings_mmap = encode_chunks_with_checkpoint(
    texts=prefixed_texts,
    model=embedding_model,
    output_npy_path=EMB_NPY_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    batch_size=batch_size,
    checkpoint_every=checkpoint_every,
)

print(f"\n✅ Encoding complete. Embedding matrix shape: {embeddings_mmap.shape}")

# %% [markdown]
# ## Cell 6 — Build & Save FAISS Index

# %%
from src.embedding import build_faiss_index

if os.path.exists(FAISS_PATH):
    print(f"FAISS index already exists: {FAISS_PATH}")
    print("Loading existing index...")
    import faiss
    index = faiss.read_index(FAISS_PATH)
    print(f"✅ Loaded — {index.ntotal:,} vectors")
else:
    print("Building FAISS index...")
    index = build_faiss_index(
        npy_path=EMB_NPY_PATH,
        index_output_path=FAISS_PATH,
    )
    print(f"\n✅ FAISS index built and saved — {index.ntotal:,} vectors")

print(f"Index type : {type(index).__name__}")
print(f"Dimension  : {index.d}")

# %% [markdown]
# ## Cell 7 — Align & Save Metadata

# %%
from src.embedding import align_and_save_metadata

meta_path    = os.path.join(IDX_DIR, 'metadata.parquet')
ids_path     = os.path.join(IDX_DIR, 'chunk_ids.json')

if os.path.exists(meta_path) and os.path.exists(ids_path):
    print(f"Metadata files already exist:")
    print(f"  {meta_path}")
    print(f"  {ids_path}")
else:
    ids_path, meta_path = align_and_save_metadata(
        df_chunks=df_chunks,
        output_dir=IDX_DIR,
    )

print("\nVerifying alignment...")
import faiss
idx = faiss.read_index(FAISS_PATH)
df_meta = pd.read_parquet(meta_path)
with open(ids_path) as f:
    ids = json.load(f)

assert idx.ntotal == len(df_meta) == len(ids), (
    f"Alignment mismatch: index={idx.ntotal}  meta={len(df_meta)}  ids={len(ids)}"
)
print(f"✅ Alignment verified — {idx.ntotal:,} vectors | {len(df_meta):,} metadata rows")

# %% [markdown]
# ## Cell 8 — Smoke Test: Query the Index

# %%
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Reload model & index (handles session resets cleanly)
if 'embedding_model' not in dir() or embedding_model is None:
    embedding_model = SentenceTransformer(emb_cfg['model_name'], device=device)

idx      = faiss.read_index(FAISS_PATH)
df_meta  = pd.read_parquet(meta_path)

# Test query with the mandatory "query: " prefix
TEST_QUERY = "query: Lãi suất ngân hàng Việt Nam năm 2023"

query_emb = embedding_model.encode(TEST_QUERY, normalize_embeddings=True).reshape(1, -1).astype('float32')
scores, faiss_ids = idx.search(query_emb, 5)

print(f"Query: {TEST_QUERY}")
print(f"\nTop-5 results:")
for rank, (fid, score) in enumerate(zip(faiss_ids[0], scores[0]), 1):
    row = df_meta.iloc[fid]
    print(f"  #{rank}  score={score:.4f}  chunk_id={row['chunk_id']}")
    print(f"       title  : {row['title'][:70]}...")
    print(f"       year   : {row['year']}  source: {row['source']}")

print("\n✅ Smoke test passed.")

# %% [markdown]
# ## Cell 9 — Cleanup Temporary .npy File
# 
# > **Run this cell ONLY after verifying the FAISS index is correct in Cell 8.**  
# > The intermediate `.npy` embedding file is no longer needed once the FAISS index is built.  
# > Deleting it frees ~200–400 MB of Drive space per strategy.

# %%
# Safety guard: only delete if FAISS index exists and is non-empty
import faiss

idx_check = faiss.read_index(FAISS_PATH)
assert idx_check.ntotal > 0, "FAISS index is empty — do NOT delete the .npy file!"

if os.path.exists(EMB_NPY_PATH):
    npy_size_mb = os.path.getsize(EMB_NPY_PATH) / 1e6
    os.remove(EMB_NPY_PATH)
    print(f"✅ Deleted: {EMB_NPY_PATH}  ({npy_size_mb:.0f} MB freed)")
else:
    print(f"ℹ️  File already removed: {EMB_NPY_PATH}")

print(f"\nFinal outputs for strategy '{TARGET_STRATEGY}':")
for f in [FAISS_PATH, meta_path, ids_path, CHECKPOINT_PATH]:
    size = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
    print(f"  {'✅' if os.path.exists(f) else '❌'}  {f}  ({size:.1f} MB)")

print(f"\n✅ Phase 3 complete for strategy: {TARGET_STRATEGY}")
print("Confirm 'Xong' before proceeding to Phase 4 (BM25 Indexing).")


