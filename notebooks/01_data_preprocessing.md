# %% [markdown]
# # Phase 0 & Phase 1: Data Preprocessing & Deep EDA
# 
# **Pipeline**: Vietnamese Financial News RAG System — v2  
# **Sections covered**: §0.4 Model Verification · §1.1 Load & Validate · §1.2 Cleaning · §1.3 EDA (8 sections) · §1.4 EDA Report · §1.5 Idempotent Save
# 
# **How to run**:
# - **Local**: `jupyter notebook` from `implementation/`
# - **Colab**: Mount Drive first (Cell 0), then run top-to-bottom
# 
# All cells are idempotent — safe to re-run without recomputing completed work.

# %% [markdown]
# ## Cell 0 — Environment Setup

# %%
import os, sys
from pathlib import Path

# ── Detect environment ──────────────────────────────────────────────────────
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()
print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

# ── Mount Drive (Colab only) ────────────────────────────────────────────────
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = Path('/content/rag-vn-finance/implementation')
    # Clone repo if not already present
    if not REPO_ROOT.exists():
        print("Cloning repo...")
        os.system('git clone https://github.com/<your-repo>/rag-vn-finance /content/rag-vn-finance')
else:
    # Local: this notebook lives at implementation/notebooks/
    REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in dir() else Path.cwd().parent
    # Fallback: hardcode relative to notebook location
    REPO_ROOT = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())

print(f"Project root: {REPO_ROOT}")
assert REPO_ROOT.exists(), f"Project root not found: {REPO_ROOT}"

# ── Add src/ to path ────────────────────────────────────────────────────────
src_path = str(REPO_ROOT)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print(f"sys.path: {sys.path[:3]}")

# ── Install dependencies if needed (Colab) ──────────────────────────────────
if IN_COLAB:
    req_path = REPO_ROOT / 'requirements.txt'
    if req_path.exists():
        os.system(f'pip install -r {req_path} -q')
        print("Dependencies installed.")
    else:
        print("WARNING: requirements.txt not found — install manually")

print("\nCell 0 complete.")

# %% [markdown]
# ## Cell 0b — Phase 0: Model Verification (§0.4)
# 
# > Run once at Phase 0. Requires OPENROUTER_API_KEY and GROQ_API_KEY in `.env` or Colab userdata.

# %%
from src.utils import load_config, get_env, resolve_path

config = load_config(REPO_ROOT / 'configs' / 'config.yaml')
print("Config loaded.")

OPENROUTER_KEY = get_env('OPENROUTER_API_KEY')
GROQ_KEY   = get_env('GROQ_API_KEY')

# ---- VERIFY OPENROUTER ----
if not OPENROUTER_KEY:
    print("WARNING: OPENROUTER_API_KEY not found — skipping OpenRouter verification")
else:
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY
        )
        
        target_model = config['generation']['model']
        print(f"Verifying OpenRouter model: {target_model}...")
        # Lightweight call to verify connection and key
        models = client.models.list()
        available = [m.id for m in models.data]
        if target_model in available:
            print(f"\n✅ {target_model} confirmed active on OpenRouter")
        else:
            print(f"\n⚠️  {target_model} NOT found — update config.yaml generation.model")
    except ImportError:
        print("⚠️  Error: 'openai' library not installed. Check requirements.txt")
    except Exception as e:
        print(f"⚠️  OpenRouter verification failed: {e}")

# ---- VERIFY GROQ ----
if not GROQ_KEY:
    print("WARNING: GROQ_API_KEY not found — skipping Groq verification")
else:
    try:
        from groq import Groq
        client_groq = Groq(api_key=GROQ_KEY)
        models = [m.id for m in client_groq.models.list().data]
        target_groq = config['evaluation']['groq_model']
        if target_groq in models:
            print(f"✅ {target_groq} confirmed active")
        else:
            print(f"⚠️  {target_groq} not found. Available Groq models:")
            for m in models:
                print(f"   - {m}")
            print("Update config.yaml evaluation.groq_model before Phase 8.")
    except ImportError:
        print("⚠️  Lỗi: Thư viện 'groq' chưa được cài đặt. Hãy kiểm tra lại requirements.txt")
    except Exception as e:
        print(f"⚠️  Groq verification failed: {e}")

print("\nCell 0b complete.")


# %% [markdown]
# ## Cell 1 — Load & Schema Validation (§1.1)

# %%
import pandas as pd
from src.utils import load_config, resolve_path
from src.preprocessing import load_and_validate

config = load_config(REPO_ROOT / 'configs' / 'config.yaml')

raw_path = resolve_path(config['data'], 'raw_path')
# Make path absolute relative to REPO_ROOT if not absolute
if not os.path.isabs(raw_path):
    raw_path = str(REPO_ROOT / raw_path)

print(f"Loading from: {raw_path}")
df_raw = load_and_validate(raw_path)

print(f"\nShape: {df_raw.shape}")
print(f"\nColumn dtypes:\n{df_raw.dtypes}")
print(f"\nFirst 3 rows:")
display(df_raw.head(3))

# %% [markdown]
# ## Cell 2 — Cleaning Pipeline (§1.2)

# %%
from src.preprocessing import clean_pipeline

drop_log_path = resolve_path(config['data'], 'drop_log_path')
if not os.path.isabs(drop_log_path):
    drop_log_path = str(REPO_ROOT / drop_log_path)

df = clean_pipeline(df_raw.copy(), drop_log_path=drop_log_path)

print(f"\nCleaning summary:")
print(f"  Raw rows  : {len(df_raw):,}")
print(f"  Clean rows: {len(df):,}")
print(f"  Dropped   : {len(df_raw) - len(df):,}")
print(f"\nNew columns added: {[c for c in df.columns if c not in df_raw.columns]}")
print(f"\nSource distribution (derived from URL):")
print(df['source'].value_counts())
print(f"\nYear range: {df['year'].min()} – {df['year'].max()}")

# %% [markdown]
# ## Cell 3 — EDA-1: Source & Temporal Coverage (§1.3)

# %%
from src.preprocessing import eda_source_temporal
import json

plots_dir = resolve_path(config['data'], 'eda_plots_dir')
if not os.path.isabs(plots_dir):
    plots_dir = str(REPO_ROOT / plots_dir)

eda1 = eda_source_temporal(df, plots_dir)

print("EDA-1 Results:")
print(json.dumps(eda1, indent=2, ensure_ascii=False))

from IPython.display import Image, display as ipy_display
ipy_display(Image(filename=os.path.join(plots_dir, 'eda1_source_temporal.png'), width=900))

# %% [markdown]
# ## Cell 4 — EDA-2: Category Distribution (§1.3)

# %%
from src.preprocessing import eda_category

eda2 = eda_category(df, plots_dir)

print("EDA-2 Results:")
print(json.dumps(eda2, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda2_category.png'), width=900))

# %% [markdown]
# ## Cell 5 — EDA-3: Content Length Analysis (§1.3)

# %%
from src.preprocessing import eda_content_length

eda3 = eda_content_length(df, plots_dir)

print("EDA-3 Results:")
print(json.dumps(eda3, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda3_content_length.png'), width=900))

# %% [markdown]
# ## Cell 6 — EDA-4: Tag & Keyword Analysis (§1.3)

# %%
from src.preprocessing import eda_tags

eda4 = eda_tags(df, plots_dir)

print("EDA-4 Results:")
print(json.dumps(eda4, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda4_tags.png'), width=900))

# %% [markdown]
# ## Cell 7 — EDA-5: Temporal Text Patterns (§1.3)

# %%
from src.preprocessing import eda_temporal_patterns

# EDA-3 must have already run to add token_count column
if 'token_count' not in df.columns:
    df['token_count'] = df['content_token_counts'].astype(int)

eda5 = eda_temporal_patterns(df, plots_dir)

print("EDA-5 Results:")
print(json.dumps(eda5, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda5_temporal_patterns.png'), width=900))

# %% [markdown]
# ## Cell 8 — EDA-6: Content Duplication & Near-Duplicate Analysis (§1.3)
# 
# > Uses `datasketch` MinHash (threshold 0.85, 2000-article sample). Falls back to hash-prefix if unavailable.

# %%
from src.preprocessing import eda_duplication

eda6 = eda_duplication(df, plots_dir)

print("EDA-6 Results:")
print(json.dumps(eda6, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda6_duplication.png'), width=700))

# %% [markdown]
# ## Cell 9 — EDA-7: Vocabulary & Readability Analysis (§1.3)

# %%
from src.preprocessing import eda_vocabulary

eda7 = eda_vocabulary(df, plots_dir)

print("EDA-7 Results:")
print(json.dumps(eda7, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda7_vocabulary.png'), width=900))

# %% [markdown]
# ## Cell 10 — EDA-8: Cross-Variable Correlation Matrix (§1.3)

# %%
from src.preprocessing import eda_correlation

# Requires token_count and ttr columns from EDA-3 and EDA-7
if 'token_count' not in df.columns:
    df['token_count'] = df['content_token_counts'].astype(int)
if 'ttr' not in df.columns:
    df['ttr'] = df['content'].apply(
        lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
    )

eda8 = eda_correlation(df, plots_dir)

print("EDA-8 Results:")
print(json.dumps(eda8, indent=2, ensure_ascii=False))

ipy_display(Image(filename=os.path.join(plots_dir, 'eda8_correlation.png'), width=800))

# %% [markdown]
# ## Cell 11 — Assemble EDA Report (§1.4)

# %%
from src.preprocessing import assemble_eda_report, save_eda_report

eda_sections = {
    'eda1': eda1, 'eda2': eda2, 'eda3': eda3, 'eda4': eda4,
    'eda5': eda5, 'eda6': eda6, 'eda7': eda7, 'eda8': eda8
}

eda_report = assemble_eda_report(df_raw, df, eda_sections)

report_path = resolve_path(config['data'], 'eda_report_path')
if not os.path.isabs(report_path):
    report_path = str(REPO_ROOT / report_path)

save_eda_report(eda_report, report_path)

print("\nEDA Report Summary:")
for k, v in eda_report.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")

print(f"\nDesign Implications:")
for i, impl in enumerate(eda_report.get('design_implications', []), 1):
    print(f"  {i}. {impl}")

# Verify plot count
png_files = list(Path(plots_dir).glob('*.png'))
print(f"\nPNG plots saved: {len(png_files)}")
for p in sorted(png_files):
    print(f"  {p.name}")

# %% [markdown]
# ## Cell 12 — Idempotent Save (§1.5)
# 
# > Saves `cleaned.parquet` (Snappy compressed). Loads from cache if already exists.

# %%
from src.preprocessing import save_cleaned

processed_path = resolve_path(config['data'], 'processed_path')
if not os.path.isabs(processed_path):
    processed_path = str(REPO_ROOT / processed_path)

df = save_cleaned(df, processed_path)

# Verify output
parquet_size_mb = os.path.getsize(processed_path) / 1e6
print(f"\nVerification:")
print(f"  File   : {processed_path}")
print(f"  Size   : {parquet_size_mb:.1f} MB  (expected 15–25 MB)")
print(f"  Rows   : {len(df):,}")
print(f"  Columns: {df.columns.tolist()}")

# Confirm drop log exists
drop_log_path = resolve_path(config['data'], 'drop_log_path')
if not os.path.isabs(drop_log_path):
    drop_log_path = str(REPO_ROOT / drop_log_path)
if os.path.exists(drop_log_path):
    drop_df = pd.read_csv(drop_log_path)
    print(f"  Drop log: {len(drop_df)} entries")

print("\n✅ Phase 1 complete. Confirm 'Xong' before proceeding to Phase 2 (Chunking).")


