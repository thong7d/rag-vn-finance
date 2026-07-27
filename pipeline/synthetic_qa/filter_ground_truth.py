import json
import math
import pandas as pd
from pathlib import Path

def filter_qa_to_target():
    # 1. Define system file paths
    qa_path = Path("synthetic_qa/qa_dataset_stratified.jsonl")
    metadata_path = Path("indexes/fixed_size/metadata.parquet")
    output_path = Path("synthetic_qa/ground_truth_final.jsonl")

    if not qa_path.exists():
        raise FileNotFoundError(f"Source QA dataset not found at: {qa_path}")

    # 2. Load all 863 QA pairs into a DataFrame
    qa_list = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_list.append(json.loads(line))
    df_qa = pd.DataFrame(qa_list)
    total_current_qa = len(df_qa)
    print(f"[INFO] Total input QA pairs: {total_current_qa}")

    # 3. Load metadata to retrieve stratum label attributes
    df_meta = pd.read_parquet(metadata_path)[["doc_id", "year", "source", "category"]].drop_duplicates(subset="doc_id")
    df_meta["year"] = df_meta["year"].fillna("Unknown").astype(str)
    df_meta["source"] = df_meta["source"].fillna("Unknown").astype(str)
    df_meta["category"] = df_meta["category"].fillna("Unknown").astype(str)

    # 4. Merge stratum labels into the QA dataset
    df_merged = df_qa.merge(df_meta, on="doc_id", how="left")

    # 5. Compute proportional allocation for the 150-sample target
    total_target = 150
    strata_cols = ["year", "source", "category"]
    
    strata_counts = df_merged.groupby(strata_cols).size().reset_index(name="pop")
    strata_counts["allocated"] = (strata_counts["pop"] / total_current_qa * total_target).apply(math.floor)

    # Round-up correction for floor-rounding remainder
    diff = total_target - strata_counts["allocated"].sum()
    if diff > 0:
        for idx in strata_counts.sort_values(by="pop", ascending=False).index[:diff]:
            strata_counts.loc[idx, "allocated"] += 1

    # 6. Draw random samples per stratum using a fixed seed (random_state=42)
    sampled_frames = []
    for _, row in strata_counts.iterrows():
        n_samples = int(row["allocated"])
        if n_samples == 0:
            continue
            
        cond = (df_merged["year"] == row["year"]) & \
               (df_merged["source"] == row["source"]) & \
               (df_merged["category"] == row["category"])
               
        stratum_df = df_merged[cond]
        sampled = stratum_df.sample(n=min(n_samples, len(stratum_df)), random_state=42)
        sampled_frames.append(sampled)

    df_final = pd.concat(sampled_frames, ignore_index=True)

    # 7. Write output file in Ragas-compatible format
    df_output = df_final[["question", "ground_truth", "contexts", "doc_id"]]
    with open(output_path, "w", encoding="utf-8") as out_f:
        for _, row in df_output.iterrows():
            out_f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Successfully filtered {len(df_output)} ground truth entries saved to: {output_path}")

if __name__ == "__main__":
    filter_qa_to_target()