# Vietnamese Financial News RAG System — Project Pipeline (v3)

---

## Project Metadata

| Item | Detail |
|------|--------|
| Course | Text Data Mining & Applications |
| Dataset | Vietnamese Financial News 2015–2024 (10K working set) |
| Domain | Vietnamese Financial News |
| Team Size | 3 members |
| Duration | 10 weeks |
| Output Language | **English** (all code, commits, filenames, report) |

### Changelog from v2

| Section | Change |
|---------|--------|
| Global | **Gemini removed entirely**. Generation model → `google/gemma-4-31b-it:free` via **OpenRouter** gateway. API library: `openai`. Env var: `OPENROUTER_API_KEY`. |
| Phase 1 | `cleaned.parquet` now includes 4 Semantic Enrichment columns: `tickers`, `is_historical`, `numerical_density`, `entities` (via `underthesea`). EDA expanded to **9 sections** (+EDA-9 Niche Glossary). |
| Phase 5 & 7 | All generation calls use OpenAI SDK pointed at `https://openrouter.ai/api/v1`. Must strip markdown fences from LLM output before `json.loads()`. |
| `config.yaml` | `generation.model` → `google/gemma-4-31b-it:free`. `evaluation.groq_model` → `llama-3.3-70b-versatile`. |
| `requirements.txt` | Removed `google-generativeai`. Added `underthesea`. `openai` already present. |

---

## Team Role Assignment

| Role | Member | Primary Phases |
|------|--------|----------------|
| Data & Indexing Engineer | Member A | Phase 0, 1, 2, 4 |
| ML & Retrieval Engineer | Member B | Phase 3, 6 |
| Generation, Eval & UI Engineer | Member C | Phase 5, 7, 8, 9 |
| Report | **All** | Phase 10 |

---

## Google Drive Folder Structure

```
MyDrive/
└── rag-vn-finance/
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   │   ├── cleaned.parquet
    │   │   ├── eda_report.json
    │   │   └── eda_plots/          # 9 PNG exports
    │   └── chunks/
    │       ├── fixed_size/
    │       ├── sentence_aware/
    │       └── article_level/
    ├── embeddings/
    ├── indexes/
    ├── bm25/
    ├── synthetic_qa/
    ├── evaluation/
    ├── checkpoints/
    └── ui/
```

---

## Global Rules

1. **Idempotent cells**: `if not os.path.exists(output_path)` before every write.
2. **Session reset protocol**: Remount Drive → reload checkpoint → verify file exists before any write cell.
3. **Checkpoint for long tasks**: Any task >15 min must checkpoint every N steps.
4. **API key security**: `.env` locally, `google.colab.userdata.get()` on Colab. Never hardcode.
5. **English only**: All filenames, code, comments, commits, report.
6. **Confirmation gate**: Confirm "Xong" before proceeding to the next phase.
7. **Bug fix protocol**: When patching any cell, re-read all preceding cells for conflicts.

---

## Colab Account Allocation

| Account | Assigned Task | Runtime |
|---------|--------------|---------|
| Account 1 | Phase 3 — Embed `fixed_size` | **GPU (T4)** |
| Account 2 | Phase 3 — Embed `sentence_aware` | **GPU (T4)** |
| Account 3 | Phase 3 — Embed `article_level` | **GPU (T4)** |
| Account 4 | Phase 5 — Synthetic QA generation | CPU |
| Account 5 | Phase 8 — Evaluation (Groq calls) | CPU |

---

## Phase 0: Project Setup & Infrastructure ✅ COMPLETED

- Repository structure created per §0.1.
- `requirements.txt` pinned (no `google-generativeai`, has `openai`, `underthesea`).
- `configs/config.yaml` with dual-path support (local / Colab).
- `.env.example` with `OPENROUTER_API_KEY` and `GROQ_API_KEY`.

### 0.4 Model Verification (Cell 0b)

```python
from openai import OpenAI

# ---- Verify OpenRouter ----
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=get_env('OPENROUTER_API_KEY')
)
models = client.models.list()
available = [m.id for m in models.data]
target = config['generation']['model']  # "google/gemma-4-31b-it:free"
assert target in available, f"❌ {target} NOT available"
print(f"✅ {target} confirmed active on OpenRouter")

# ---- Verify Groq ----
from groq import Groq
client_groq = Groq(api_key=get_env('GROQ_API_KEY'))
groq_models = [m.id for m in client_groq.models.list().data]
target_groq = config['evaluation']['groq_model']
assert target_groq in groq_models, f"❌ {target_groq} NOT available"
print(f"✅ {target_groq} confirmed active")
```

---

## Phase 1: Data Ingestion, Preprocessing & Deep EDA ✅ COMPLETED

### Output
- `data/processed/cleaned.parquet` — 9,999 rows, ~24.6 MB
- `data/processed/eda_report.json` — 9 populated sections
- `data/processed/eda_plots/*.png` — 9 PNG plots
- `data/processed/drop_log.csv` — 1 row dropped

### cleaned.parquet Schema (16 columns)

| Column | Type | Source |
|--------|------|--------|
| `url` | str | raw |
| `title` | str | raw (cleaned) |
| `time` | datetime | parsed from raw |
| `category` | str | raw |
| `content` | str | raw (cleaned) |
| `tags` | str | raw (cleaned) |
| `content_token_counts` | int | raw |
| `year` | int | derived |
| `month` | int | derived |
| `yearmonth` | str | derived |
| `doc_id` | str | sha256(url)[:16] |
| `source` | str | derived from URL domain |
| **`tickers`** | str (JSON list) | **Semantic Enrichment** — regex from content+title |
| **`is_historical`** | bool | **Semantic Enrichment** — year < 2020 |
| **`numerical_density`** | float | **Semantic Enrichment** — digit ratio |
| **`entities`** | str (comma-sep) | **Semantic Enrichment** — underthesea NER (LOC, ORG, PER) |

### EDA Sections (9 total)
1. EDA-1: Source & Temporal Coverage
2. EDA-2: Category Distribution
3. EDA-3: Content Length Analysis
4. EDA-4: Tag & Keyword Analysis
5. EDA-5: Temporal Text Patterns
6. EDA-6: Content Duplication & Near-Duplicate Analysis
7. EDA-7: Vocabulary & Readability Analysis
8. EDA-8: Cross-Variable Correlation Matrix
9. **EDA-9: Niche Glossary** — articles containing definition patterns

---

## Phase 2: Chunking Strategy Experiments

**Duration**: Week 2 | **Owner**: Member A | **Hardware**: Colab CPU

### Purpose
Split cleaned articles into retrieval-ready chunks using 3 strategies. Every chunk must carry full article metadata for downstream filtering.

### Input
- `data/processed/cleaned.parquet`

### Output (per strategy)
- `data/chunks/{strategy}/chunks.parquet`
- `data/chunks/{strategy}/stats.json`

### Tokenizer
All token counts use `transformers.AutoTokenizer` with model `intfloat/multilingual-e5-large` (same model used in Phase 3 embedding).

### 3 Strategies

#### Strategy 1: `fixed_size` (256 tokens, 32 overlap)
- Tokenize `content` with the E5 tokenizer.
- Sliding window: 256 tokens per chunk, 32 token overlap.
- **Title prefix**: prepend `"Title: {title}\n"` to the `text` field of each chunk.
- Expected output: ~40–50K chunks.

#### Strategy 2: `sentence_aware` (5 sentences, 1 overlap)
- Split content into sentences using `sentence-splitter`.
- Group 5 consecutive sentences per chunk with 1 sentence overlap.
- **Fallback**: if any single sentence exceeds 256 tokens, split it by token with overlap.
- Expected output: ~30–45K chunks.

#### Strategy 3: `article_level` (max 512 tokens)
- Truncate content at 512 tokens.
- 1 chunk per article (1:1 mapping).
- Expected output: ~10K chunks.

### Chunk Output Schema

Every row in `chunks.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | str | `{doc_id}_c{index:04d}` |
| `doc_id` | str | Parent article ID |
| `text` | str | Chunk text content |
| `chunk_index` | int | 0-based index within article |
| `total_chunks` | int | Total chunks for this article |
| `strategy` | str | `"fixed_size"` / `"sentence_aware"` / `"article_level"` |
| `source` | str | Article metadata |
| `category` | str | Article metadata |
| `time` | str | Article metadata |
| `year` | int | Article metadata |
| `title` | str | Article metadata |
| `url` | str | Article metadata |
| `tickers` | str | Semantic Enrichment metadata |
| `is_historical` | bool | Semantic Enrichment metadata |
| `numerical_density` | float | Semantic Enrichment metadata |
| `entities` | str | Semantic Enrichment metadata |

### Stats JSON Schema

```json
{
  "strategy": "fixed_size",
  "total_chunks": 42310,
  "total_articles": 9999,
  "avg_tokens_per_chunk": 241.3,
  "max_tokens_per_chunk": 256,
  "min_tokens_per_chunk": 18,
  "truncated_chunks": 0,
  "chunks_per_article": {"mean": 4.2, "median": 4.0, "max": 12}
}
```

### Idempotent Save
```python
output_path = f"data/chunks/{strategy}/chunks.parquet"
if not os.path.exists(output_path):
    df_chunks.to_parquet(output_path, index=False, compression="snappy")
else:
    print("Already exists. Loading from cache.")
    df_chunks = pd.read_parquet(output_path)
```

---

## Phase 3: Embedding & FAISS Indexing

**Duration**: Week 3–4 | **Owner**: Member B | **Hardware**: ⚡ Colab GPU (T4) REQUIRED

- Model: `intfloat/multilingual-e5-large` (560M params, dim=1024)
- Prefix: `"passage: "` for docs, `"query: "` for queries
- Checkpoint every 100 batches via `np.memmap` + `checkpoint.json`
- 3 accounts run in parallel (one per strategy)
- Output: `index.faiss` + `metadata.parquet` per strategy

---

## Phase 4: Sparse Indexing (BM25)

**Duration**: Week 4 | **Owner**: Member A | **Hardware**: Colab CPU

BM25 index per chunking strategy using `rank-bm25`.

---

## Phase 5: Synthetic QA Generation — Batched

**Duration**: Week 4–5 | **Owner**: Member C | **Hardware**: Colab CPU

### ⚠️ CRITICAL: OpenRouter API Pattern

```python
from openai import OpenAI
import json, re

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=get_env('OPENROUTER_API_KEY')
)

response = client.chat.completions.create(
    model=config['generation']['model'],  # "google/gemma-4-31b-it:free"
    messages=[
        {"role": "system", "content": "You are a financial analyst..."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    response_format={"type": "json_object"}
)
output_text = response.choices[0].message.content
```

### ⚠️ MANDATORY: Markdown Fence Stripping

Open-source models frequently wrap JSON output in markdown code fences. **Always strip before parsing.**

```python
def strip_markdown_json(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?\s*```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

# Usage:
raw_output = response.choices[0].message.content
clean_json = strip_markdown_json(raw_output)
qa_pairs = json.loads(clean_json)
```

### Batching Strategy

| Parameter | Value |
|-----------|-------|
| Articles per prompt | 10 |
| QA pairs per article | 2 |
| Target articles | 500 |
| **Total requests** | **50** |

### Checkpointing & Validation
- Checkpoint every 5 batches (50 articles).
- Remove QA pairs with invalid `doc_id`, short Q/A, or duplicate questions.
- Max 2 retries with exponential backoff on parse failure.

### Output Files
- `synthetic_qa/qa_pairs_raw.jsonl`
- `synthetic_qa/qa_pairs_filtered.parquet`
- `synthetic_qa/qa_pairs_train.parquet` (80%)
- `synthetic_qa/qa_pairs_test.parquet` (20%, **immutable**)
- `synthetic_qa/generation_checkpoint.json`
- `synthetic_qa/generation_stats.json`

---

## Phase 6: Retrieval Pipeline

Dense + Sparse + Hybrid RRF × 3 chunking strategies = 9 configs.
Metrics: Precision@K, Recall@K, MRR, NDCG@10.

---

## Phase 7: Generation Pipeline

**Uses OpenRouter** (same pattern as Phase 5):

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=get_env('OPENROUTER_API_KEY')
)

response = client.chat.completions.create(
    model=config['generation']['model'],
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    temperature=config['generation']['temperature'],
    max_tokens=config['generation']['max_tokens']
)
answer = response.choices[0].message.content
```

> **Note**: OpenRouter free tier has generous limits. No RPD budget constraints like the old Gemini setup.

---

## Phase 8: End-to-End Evaluation — Primary: Groq + Llama

**Duration**: Week 7–8 | **Owner**: Member C | **Hardware**: Colab CPU

- Primary evaluator: **Groq + Llama 3.3 70B** (cross-model, zero bias).
- Layer 1: Reference-free metrics (ROUGE-L, BERTScore).
- Layer 2: LLM-as-Judge (4 dimensions: Faithfulness, Answer Relevancy, Context Precision, Answer Correctness).
- Layer 3: Retrieval metrics from Phase 6.
- Output: `evaluation/final_comparison_table.csv` (9 configs × all metrics).

---

## Phase 9: UI & Demo Application

Gradio app on Hugging Face Spaces. Features: query box, strategy selector, year range slider, source citations, confidence score.

---

## Phase 10: Report & Final Presentation

### Final Deliverables
- [ ] Report (PDF, English)
- [ ] GitHub repo (public, full README + demo link)
- [ ] HF Spaces demo (public URL)
- [ ] Presentation slides (max 12 slides)
- [ ] `evaluation/final_comparison_table.csv`
- [ ] `data/processed/eda_report.json`
- [ ] `data/processed/eda_plots/` — 9 PNGs committed

---

## Timeline Summary

| Week | Phase | Owner | Key Milestone |
|------|-------|-------|---------------|
| 1 | Phase 0 | All | Repo + Drive + model verification ✅ |
| 1–2 | Phase 1 | A | cleaned.parquet + 9-section EDA ✅ |
| 2 | Phase 2 | A | 3 chunk datasets on Drive |
| 3–4 | Phase 3 | B | 3 FAISS indexes verified |
| 4 | Phase 4 | A | 3 BM25 indexes on Drive |
| 4–5 | Phase 5 | C | ~1,000 QA pairs, test set locked |
| 5–6 | Phase 6 | B | 9-config retrieval benchmark |
| 6 | Phase 7 | C | 200 generation results saved |
| 7–8 | Phase 8 | C | Full evaluation table + error analysis |
| 8–9 | Phase 9 | C+B | HF Spaces demo live |
| 9–10 | Phase 10 | All | Report + slides submitted |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| OpenRouter model unavailable | High | Verify at Phase 0; fallback to other `:free` models |
| OpenRouter rate limit hit | Medium | Add `time.sleep()` between batches; retry with backoff |
| Groq model deprecated | Medium | Verify at Phase 0; name only in `config.yaml` |
| Colab timeout mid-embedding | High | `np.memmap` + 100-batch checkpoint |
| LLM wraps JSON in markdown fences | High | **Mandatory `strip_markdown_json()` before `json.loads()`** |
| Batch prompt JSON parse failure | Medium | Max 2 retries + exponential backoff + failed_batches log |
| Test set accidentally modified | High | MD5 hash locked at Phase 5, verified at Phase 8 |
| Drive storage full | Medium | Delete `.npy` after FAISS build; Snappy Parquet |
