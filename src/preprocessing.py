"""
preprocessing.py — Data loading, cleaning, and EDA for Phase 1.

Pipeline v2 §1.1, §1.2, §1.3, §1.4, §1.5
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import numpy as np
try:
    from underthesea import ner
except (ImportError, OSError) as e:
    import logging
    logging.warning(f"Không thể tải NER model (Underthesea) do lỗi: {e}. Hệ thống sẽ gán null cho cột 'entities'.")
    ner = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

EXPECTED_COLS = ["url", "title", "time", "category", "content", "tags", "content_token_counts"]

FIN_TERMS = [
    "lãi suất", "chứng khoán", "cổ phiếu", "ngân hàng",
    "tín dụng", "đầu tư", "GDP", "lạm phát", "tỷ giá", "vốn hóa"
]


# ---------------------------------------------------------------------------
# 1.1  Load & validate
# ---------------------------------------------------------------------------

def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and validate schema (§1.1)."""
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Schema mismatch — missing columns: {missing}")
    logger.info(f"Schema OK. Shape: {df.shape}")
    logger.info(f"Dtypes:\n{df.dtypes}")
    return df


# ---------------------------------------------------------------------------
# 1.2  Cleaning pipeline (7 steps, exact order)
# ---------------------------------------------------------------------------

def clean_pipeline(df: pd.DataFrame, drop_log_path: str) -> pd.DataFrame:
    """
    Execute the 7-step cleaning pipeline (§1.2).
    Logs every dropped row to drop_log_path.
    """
    drop_records = []
    initial_count = len(df)

    # Step 1: Drop exact-duplicate rows on url
    dup_mask = df.duplicated(subset=["url"], keep="first")
    for idx in df[dup_mask].index:
        drop_records.append({"original_index": idx, "reason": "duplicate_url",
                              "url": df.at[idx, "url"]})
    df = df[~dup_mask].copy()
    logger.info(f"Step 1 — Dropped {dup_mask.sum()} duplicate URLs")

    # Step 2: Drop rows where content is null or too short
    null_mask = df["content"].isna()
    short_mask = df["content"].fillna("").apply(lambda x: len(x.strip()) < 100)
    bad_content = null_mask | short_mask
    for idx in df[bad_content].index:
        reason = "null_content" if null_mask.get(idx, False) else "short_content"
        drop_records.append({"original_index": idx, "reason": reason,
                              "url": df.at[idx, "url"]})
    df = df[~bad_content].copy()
    logger.info(f"Step 2 — Dropped {bad_content.sum()} rows (null/short content)")

    # Step 3: Parse time -> datetime; extract year, month, yearmonth
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    bad_time = df["time"].isna()
    for idx in df[bad_time].index:
        drop_records.append({"original_index": idx, "reason": "unparseable_time",
                              "url": df.at[idx, "url"]})
    df = df[~bad_time].copy()
    df["year"] = df["time"].dt.year.astype(int)
    df["month"] = df["time"].dt.month.astype(int)
    df["yearmonth"] = df["time"].dt.to_period("M").astype(str)
    logger.info(f"Step 3 — Parsed time column, dropped {bad_time.sum()} unparseable rows")

    # Step 4: Normalize whitespace in content, title, tags
    for col in ["content", "title", "tags"]:
        df[col] = df[col].fillna("").apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())

    # Step 5: Fill null tags with ""
    df["tags"] = df["tags"].fillna("")

    # Step 6: Add doc_id = sha256(url)[:16]
    df["doc_id"] = df["url"].apply(lambda u: hashlib.sha256(u.encode()).hexdigest()[:16])

    # Step 7: Derive source from url domain
    def _extract_source(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            return host.replace("www.", "")
        except Exception:
            return "unknown"

    df["source"] = df["url"].apply(_extract_source)

    # Save drop log
    ensure_dir(Path(drop_log_path).parent)
    drop_df = pd.DataFrame(drop_records)
    drop_df.to_csv(drop_log_path, index=False, encoding="utf-8")
    logger.info(f"Drop log saved to {drop_log_path} — total dropped: {len(drop_records)}/{initial_count}")

    # YÊU CẦU 1: Semantic Enrichment
    logger.info("Applying Semantic Enrichment...")
    
    # [Ticker Extraction]
    def extract_tickers(text):
        if not isinstance(text, str): return []
        patterns = [
            r'\(([A-Z]{3})\)',
            r'\[([A-Z]{3})\]',
            r'(?:mã CK|mã ck|mã Ck):?\s*([A-Z]{3})',
            r'(?:cổ phiếu|mã)\s+([A-Z]{3})'
        ]
        tickers = []
        for p in patterns:
            tickers.extend(re.findall(p, text))
        return list(set(tickers))

    df['tickers'] = df.apply(lambda row: json.dumps(list(set(extract_tickers(str(row.get('content', ''))) + extract_tickers(str(row.get('title', '')))))), axis=1)

    # [Staleness Flag]
    df['is_historical'] = df['year'] < 2020

    # [Numerical Density]
    def calc_num_density(text):
        text = str(text)
        if not text: return 0.0
        digits = sum(c.isdigit() for c in text)
        return digits / max(len(text), 1)
    
    df['numerical_density'] = df['content'].apply(calc_num_density)

    # [NER Entities]
    def extract_ner(text):
        if not isinstance(text, str) or not text.strip(): return ""
        if 'ner' not in globals() or ner is None: return ""
        text = text[:1000]
        try:
            results = ner(text)
            entities = []
            for word, pos, chunk, label in results:
                if label != 'O':
                    ent_type = label.split('-')[-1]
                    if ent_type in ['LOC', 'ORG', 'PER']:
                        entities.append(word.replace('_', ' '))
            return ",".join(list(set(entities)))
        except Exception:
            return ""

    df['entities'] = df['content'].apply(extract_ner)

    df = df.reset_index(drop=True)
    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 1.3  EDA helper — save plot
# ---------------------------------------------------------------------------

def _save_plot(fig: plt.Figure, plots_dir: str, name: str) -> str:
    ensure_dir(plots_dir)
    path = os.path.join(plots_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {path}")
    return path


# ---------------------------------------------------------------------------
# EDA-1: Source & Temporal Coverage
# ---------------------------------------------------------------------------

def eda_source_temporal(df: pd.DataFrame, plots_dir: str) -> dict:
    source_dist = df["source"].value_counts().to_dict()
    year_dist = df["year"].value_counts().sort_index().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("EDA-1: Source & Temporal Coverage", fontsize=14, fontweight="bold")

    # Stacked bar: articles per year by source
    year_source = df.groupby(["year", "source"]).size().unstack(fill_value=0)
    year_source.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10")
    axes[0].set_title("Articles per Year by Source")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Article Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Source", fontsize=8)

    # Heatmap: year x month
    pivot = df.groupby(["year", "month"]).size().unstack(fill_value=0)
    sns.heatmap(pivot, ax=axes[1], cmap="YlOrRd", annot=True, fmt="d",
                linewidths=0.5, cbar_kws={"label": "Article Count"})
    axes[1].set_title("Article Volume Heatmap (Year × Month)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Year")

    _save_plot(fig, plots_dir, "eda1_source_temporal.png")

    dead_months = int((df.groupby("yearmonth").size() < 10).sum())
    return {
        "source_distribution": source_dist,
        "year_distribution": {str(k): int(v) for k, v in year_dist.items()},
        "dead_months_lt10_articles": dead_months,
        "design_implication": "QA generation in Phase 5 must stratify by year AND source to avoid 2021-2024 overrepresentation."
    }


# ---------------------------------------------------------------------------
# EDA-2: Category Distribution
# ---------------------------------------------------------------------------

def eda_category(df: pd.DataFrame, plots_dir: str) -> dict:
    cat_dist = df["category"].value_counts().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("EDA-2: Category Distribution", fontsize=14, fontweight="bold")

    # Horizontal bar
    cat_series = df["category"].value_counts()
    cat_series.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Category Frequency")
    axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()

    # Stacked area: category share over time
    cat_year = df.groupby(["year", "category"]).size().unstack(fill_value=0)
    cat_year_pct = cat_year.div(cat_year.sum(axis=1), axis=0)
    cat_year_pct.plot(kind="area", ax=axes[1], colormap="tab20", alpha=0.7)
    axes[1].set_title("Category Share Over Time")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))

    _save_plot(fig, plots_dir, "eda2_category.png")

    under_rep = [c for c, n in cat_dist.items() if n / len(df) < 0.02]
    return {
        "category_distribution": {k: int(v) for k, v in cat_dist.items()},
        "under_represented_categories": under_rep,
        "design_implication": "Retrieval evaluation should report metrics per category to detect per-domain performance gaps."
    }


# ---------------------------------------------------------------------------
# EDA-3: Content Length Analysis
# ---------------------------------------------------------------------------

def eda_content_length(df: pd.DataFrame, plots_dir: str) -> dict:
    df["token_count"] = df["content_token_counts"].astype(int)
    df["word_count"] = df["content"].apply(lambda x: len(x.split()))
    df["char_count"] = df["content"].apply(len)

    tc = df["token_count"]
    p50, p90, p99 = float(tc.median()), float(tc.quantile(0.90)), float(tc.quantile(0.99))
    pct_over_512 = float((tc > 512).mean())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("EDA-3: Content Length Analysis", fontsize=14, fontweight="bold")

    # Box plot by category
    cats = df["category"].value_counts().index[:10].tolist()
    data_by_cat = [df[df["category"] == c]["token_count"].values for c in cats]
    axes[0].boxplot(data_by_cat, labels=cats, vert=True)
    axes[0].set_title("Token Count by Category (top 10)")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Token Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Histogram with percentile lines
    axes[1].hist(tc, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    for p, label, color in [(p50, "p50", "green"), (p90, "p90", "orange"), (p99, "p99", "red")]:
        axes[1].axvline(p, color=color, linestyle="--", linewidth=1.5, label=f"{label}={p:.0f}")
    axes[1].set_title("Token Count Distribution")
    axes[1].set_xlabel("Token Count")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    _save_plot(fig, plots_dir, "eda3_content_length.png")

    return {
        "token_length": {"mean": float(tc.mean()), "median": p50, "p90": p90, "p99": p99},
        "articles_exceeding_512_tokens_pct": round(pct_over_512, 4),
        "design_implication": "article_level chunking will truncate ~{:.0%} of articles — document as main weakness.".format(pct_over_512)
    }


# ---------------------------------------------------------------------------
# EDA-4: Tag & Keyword Analysis
# ---------------------------------------------------------------------------

def eda_tags(df: pd.DataFrame, plots_dir: str) -> dict:
    all_tags = []
    for tag_str in df["tags"]:
        tags = [t.strip() for t in str(tag_str).split(",") if t.strip()]
        all_tags.extend(tags)

    tag_freq = Counter(all_tags)
    top_tags = [t for t, _ in tag_freq.most_common(30)]
    top_50_tags = [t for t, _ in tag_freq.most_common(50)]
    has_tag = df["tags"].apply(lambda x: len(str(x).strip()) > 0).mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("EDA-4: Tag & Keyword Analysis", fontsize=14, fontweight="bold")

    # Bar chart top 30
    top30 = tag_freq.most_common(30)
    tags_list = [t for t, _ in top30]
    counts_list = [c for _, c in top30]
    axes[0].barh(tags_list[::-1], counts_list[::-1], color="teal")
    axes[0].set_title("Top 30 Tags")
    axes[0].set_xlabel("Frequency")

    # WordCloud
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=600, height=400, background_color="white",
                       font_path=None, max_words=100)
        wc.generate_from_frequencies(tag_freq)
        axes[1].imshow(wc, interpolation="bilinear")
        axes[1].axis("off")
        axes[1].set_title("Tag Word Cloud")
    except Exception:
        axes[1].text(0.5, 0.5, "WordCloud unavailable", ha="center", va="center")
        axes[1].axis("off")

    _save_plot(fig, plots_dir, "eda4_tags.png")

    return {
        "tag_coverage_pct": round(float(has_tag), 4),
        "unique_tags": len(tag_freq),
        "top_10_tags": [t for t, _ in tag_freq.most_common(10)],
        "top_50_tags_for_qa_sampling": top_50_tags,
        "design_implication": "Top 50 tags can serve as seed terms for generating diverse evaluation queries in Phase 5."
    }


# ---------------------------------------------------------------------------
# EDA-5: Temporal Text Patterns
# ---------------------------------------------------------------------------

def eda_temporal_patterns(df: pd.DataFrame, plots_dir: str) -> dict:
    length_trend = df.groupby("yearmonth")["token_count"].mean().reset_index()
    monthly_volume = df.groupby("month")["doc_id"].count()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("EDA-5: Temporal Text Patterns", fontsize=14, fontweight="bold")

    axes[0].plot(range(len(length_trend)), length_trend["token_count"], color="steelblue", linewidth=1.2)
    tick_step = max(1, len(length_trend) // 12)
    axes[0].set_xticks(range(0, len(length_trend), tick_step))
    axes[0].set_xticklabels(length_trend["yearmonth"][::tick_step], rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Average Token Count Over Time")
    axes[0].set_xlabel("Year-Month")
    axes[0].set_ylabel("Mean Token Count")

    monthly_volume.plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Article Volume by Month (Seasonality)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Article Count")
    axes[1].tick_params(axis="x", rotation=0)

    _save_plot(fig, plots_dir, "eda5_temporal_patterns.png")

    anomalous_months = df.groupby("yearmonth").size()
    flagged = anomalous_months[anomalous_months < 5].index.tolist()

    return {
        "anomalous_yearmonths_lt5_articles": flagged,
        "design_implication": "Months with <5 articles may produce misleading low-recall retrieval scores — flag in evaluation notes."
    }


# ---------------------------------------------------------------------------
# EDA-6: Content Duplication & Near-Duplicate Analysis
# ---------------------------------------------------------------------------

def eda_duplication(df: pd.DataFrame, plots_dir: str) -> dict:
    exact_dups = int(df.duplicated(subset=["content"], keep="first").sum())

    # Near-duplicate via MinHash (sample 2000 articles, threshold 0.85)
    near_dup_count = 0
    sim_scores = []
    method_used = "minhash"

    try:
        from datasketch import MinHash, MinHashLSH

        sample = df.sample(min(2000, len(df)), random_state=42)
        lsh = MinHashLSH(threshold=0.85, num_perm=128)
        minhashes = {}

        for idx, row in sample.iterrows():
            m = MinHash(num_perm=128)
            for word in str(row["content"]).lower().split():
                m.update(word.encode("utf8"))
            key = str(row["doc_id"])
            minhashes[key] = m
            try:
                lsh.insert(key, m)
            except ValueError:
                pass  # duplicate key in LSH

        # Count near-duplicates
        near_dup_pairs = set()
        for key, m in minhashes.items():
            neighbors = lsh.query(m)
            for n in neighbors:
                if n != key:
                    pair = tuple(sorted([key, n]))
                    near_dup_pairs.add(pair)

        near_dup_count = len(near_dup_pairs)

        # Sample similarities for histogram
        rng = np.random.default_rng(42)
        keys = list(minhashes.keys())
        pairs_sample = [(keys[i], keys[j])
                        for i, j in zip(rng.integers(0, len(keys), 200),
                                        rng.integers(0, len(keys), 200))
                        if i != j]
        sim_scores = [minhashes[a].jaccard(minhashes[b]) for a, b in pairs_sample[:200]]

    except ImportError:
        method_used = "hash_fallback"
        logger.warning("datasketch not available — using hash fallback for near-duplicate detection")
        prefix_hash = df["content"].apply(lambda x: hashlib.md5(str(x)[:200].encode()).hexdigest())
        near_dup_count = int(prefix_hash.duplicated(keep="first").sum())
        sim_scores = []

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("EDA-6: Near-Duplicate Similarity Distribution", fontsize=14, fontweight="bold")
    if sim_scores:
        ax.hist(sim_scores, bins=30, color="purple", edgecolor="white", alpha=0.8)
        ax.axvline(0.85, color="red", linestyle="--", label="Threshold 0.85")
        ax.set_xlabel("Jaccard Similarity")
        ax.set_ylabel("Pair Count")
        ax.set_title(f"Pairwise Similarity Scores (method: {method_used})")
        ax.legend()
    else:
        ax.text(0.5, 0.5, f"Similarity histogram not available\n(method: {method_used})",
                ha="center", va="center")
    _save_plot(fig, plots_dir, "eda6_duplication.png")

    near_dup_pct = near_dup_count / max(len(df), 1)
    return {
        "exact_duplicates_on_content": exact_dups,
        "near_duplicates_found": near_dup_count,
        "near_dup_detection_method": method_used,
        "near_dup_pct": round(near_dup_pct, 4),
        "design_implication": (
            "Add near-duplicate flag to chunk metadata (is_near_dup: bool) to penalize repetitive retrieval."
            if near_dup_pct > 0.05 else
            "Near-duplicate rate <5% — no additional deduplication step needed."
        )
    }


# ---------------------------------------------------------------------------
# EDA-7: Vocabulary & Readability Analysis
# ---------------------------------------------------------------------------

def eda_vocabulary(df: pd.DataFrame, plots_dir: str) -> dict:
    def _get_vocab(texts, min_freq=5):
        tokens = []
        for t in texts:
            tokens.extend(re.findall(r"\b\w+\b", str(t).lower()))
        return Counter(tokens)

    vocab = _get_vocab(df["content"])
    vocab_size = len([w for w, c in vocab.items() if c >= 5])

    df["ttr"] = df["content"].apply(
        lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
    )

    fin_term_coverage = {}
    for term in FIN_TERMS:
        col = f"has_{term.replace(' ', '_')}"
        df[col] = df["content"].str.contains(term, case=False, na=False)
        fin_term_coverage[term] = round(float(df[col].mean()), 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("EDA-7: Vocabulary & Readability", fontsize=14, fontweight="bold")

    # Zipf plot
    sorted_freqs = sorted(vocab.values(), reverse=True)[:5000]
    axes[0].loglog(range(1, len(sorted_freqs) + 1), sorted_freqs, color="navy", linewidth=1)
    axes[0].set_title("Zipf Distribution of Word Frequencies")
    axes[0].set_xlabel("Rank (log)")
    axes[0].set_ylabel("Frequency (log)")

    # Financial term coverage
    terms = list(fin_term_coverage.keys())
    covs = list(fin_term_coverage.values())
    axes[1].barh(terms, covs, color="darkorange")
    axes[1].axvline(0.20, color="red", linestyle="--", label="20% threshold")
    axes[1].set_title("Financial Term Coverage")
    axes[1].set_xlabel("Proportion of Articles")
    axes[1].legend()

    _save_plot(fig, plots_dir, "eda7_vocabulary.png")

    safe_terms = [t for t, c in fin_term_coverage.items() if c >= 0.20]
    return {
        "vocab_size_min_freq5": vocab_size,
        "financial_term_coverage": fin_term_coverage,
        "safe_query_terms_gt20pct": safe_terms,
        "design_implication": "Financial terms with <5% coverage are risky for retrieval evaluation; use high-coverage terms as QA anchors."
    }


# ---------------------------------------------------------------------------
# EDA-8: Cross-Variable Correlation Matrix
# ---------------------------------------------------------------------------

def eda_correlation(df: pd.DataFrame, plots_dir: str) -> dict:
    numeric_cols = ["token_count", "year", "month", "ttr"]
    fin_cols = [f"has_{t.replace(' ', '_')}" for t in FIN_TERMS
                if f"has_{t.replace(' ', '_')}" in df.columns]
    numeric_cols += fin_cols

    available = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("EDA-8: Cross-Variable Correlation Matrix", fontsize=14, fontweight="bold")
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, annot_kws={"size": 7})
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _save_plot(fig, plots_dir, "eda8_correlation.png")

    return {
        "correlation_matrix_shape": list(corr_matrix.shape),
        "features_included": available,
        "design_implication": "Correlated financial terms can build multi-topic evaluation queries that test multi-concept retrieval."
    }


# ---------------------------------------------------------------------------
# EDA-9: Niche Glossary
# ---------------------------------------------------------------------------

def eda_niche_glossary(df: pd.DataFrame, plots_dir: str) -> dict:
    def_keywords = ["có nghĩa là", "được hiểu là", "là thuật ngữ", "khái niệm"]
    
    def contains_def(text):
        text = str(text).lower()
        return any(k in text for k in def_keywords)
        
    has_definition = df['content'].apply(contains_def)
    def_articles = df[has_definition]
    def_count = len(def_articles)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("EDA-9: Articles with Definitions by Category", fontsize=14, fontweight="bold")
    
    if def_count > 0:
        cat_counts = def_articles['category'].value_counts()
        cat_counts.plot(kind='bar', ax=ax, color='mediumseagreen')
        ax.set_title("Count of Definition Articles")
        ax.set_xlabel("Category")
        ax.set_ylabel("Article Count")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No definition articles found", ha="center", va="center")
        
    _save_plot(fig, plots_dir, "eda9_niche_glossary.png")
    
    return {
        "definition_articles_count": int(def_count),
        "design_implication": "Articles containing definitions can be leveraged for terminology-specific QA pairs."
    }


# ---------------------------------------------------------------------------
# 1.4  Assemble EDA report
# ---------------------------------------------------------------------------

def assemble_eda_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                        eda_sections: dict) -> dict:
    """Combine all EDA section results into the §1.4 report schema."""
    report = {
        "total_documents_raw": len(df_raw),
        "total_documents_clean": len(df_clean),
        "dropped_rows": len(df_raw) - len(df_clean),
        **eda_sections.get("eda1", {}),
        **eda_sections.get("eda2", {}),
        **eda_sections.get("eda3", {}),
        **eda_sections.get("eda4", {}),
        **eda_sections.get("eda5", {}),
        **eda_sections.get("eda6", {}),
        **eda_sections.get("eda7", {}),
        **eda_sections.get("eda8", {}),
        **eda_sections.get("eda9", {}),
        "design_implications": [
            v["design_implication"]
            for k, v in eda_sections.items()
            if "design_implication" in v
        ]
    }
    # Remove individual design_implication keys to avoid duplication
    for key in list(report.keys()):
        if key == "design_implication":
            del report[key]
    return report


# ---------------------------------------------------------------------------
# 1.5  Idempotent save
# ---------------------------------------------------------------------------

def save_cleaned(df: pd.DataFrame, processed_path: str) -> pd.DataFrame:
    """Save cleaned parquet (idempotent). Load from cache if already exists."""
    if not os.path.exists(processed_path):
        ensure_dir(Path(processed_path).parent)
        df.to_parquet(processed_path, index=False, compression="snappy")
        logger.info(f"Saved cleaned data: {len(df)} rows → {processed_path}")
    else:
        logger.info(f"Cached parquet found. Loading from: {processed_path}")
        df = pd.read_parquet(processed_path)
    return df


def save_eda_report(report: dict, report_path: str) -> None:
    """Save EDA report JSON (always overwrite — report may be refined)."""
    ensure_dir(Path(report_path).parent)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"EDA report saved: {report_path}")
