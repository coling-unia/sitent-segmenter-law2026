"""
summarise_results.py — Summarise evaluation results
---------------------------------

Input:
    Data/evaluation_results_model.csv

Output:
    Data/model_exact_match_table.csv
    A summary table of exact match percentages between annotators and the model, averaged over documents, as well as the average exact match percentage for the union of all annotators vs the model.
    
    Usage:
    python summarise_results.py [--root <path>]
"""


import argparse
from pathlib import Path
import pandas as pd



def derive_condition(row: pd.Series) -> str:
    lang = str(row["language"]).upper()
    doc  = str(row["document"]).upper()

    if "DE-HIST" in doc or "DE-HIST" in lang:
        return "DE-HIST"
    if "DE" in lang or "DE-CURRENT" in doc:
        return "DE-CURRENT"
    if "E-HIST" in lang or "E-HIST" in doc:
        return "E-HIST"
    return "E-CURRENT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    csv_in  = args.root / "Data" / "evaluation_results_model.csv"
    csv_out = args.root / "Data" / "model_exact_match_table.csv"

    df = pd.read_csv(csv_in)
    df["condition"] = df.apply(derive_condition, axis=1)

    dup = df["document"].str.contains("anthus_part_1_veronika", na=False)
    if dup.sum():
        print(f"Dropping {dup.sum()} duplicate row(s) (anthus_part_1_veronika).")
    df = df[~dup].copy()

    print("\nCondition assignments:")
    print(df[["document", "condition"]].drop_duplicates().to_string(index=False))
    print()

    MODEL = {"model"}
    df["a_lower"] = df["annotator_a"].str.lower()
    df["b_lower"] = df["annotator_b"].str.lower()

    hm = df[
        df["a_lower"].isin(MODEL) | df["b_lower"].isin(MODEL)
    ].copy()

    swap = hm["a_lower"].isin(MODEL)
    for cols in [
        ("annotator_a",  "annotator_b"),
        ("spans_a",      "spans_b"),
        ("exact_pct_a",  "exact_pct_b"),
    ]:
        hm.loc[swap, list(cols)] = hm.loc[swap, list(reversed(cols))].values

    hm["a_lower"] = hm["annotator_a"].str.lower()  # refresh after swap


    ann_doc = (
        hm.groupby(["condition", "annotator_a", "document"])["exact_pct_a"]
          .mean()
          .reset_index()
    )
    ann_summary = (
        ann_doc.groupby(["condition", "annotator_a"])
               .agg(avg_exact_pct=("exact_pct_a", "mean"))
               .reset_index()
               .sort_values(["condition", "annotator_a"])
    )

    print("Per-annotator exact match % vs Model (averaged over documents):")
    print(ann_summary.to_string(index=False))
    print()

    doc_union = (
        hm.groupby(["condition", "document"])["exact_pct_a"]
          .mean()
          .reset_index()
          .rename(columns={"exact_pct_a": "union_exact_pct"})
    )
    union_summary = (
        doc_union.groupby("condition")
                 .agg(
                     n_docs              = ("document",         "count"),
                     avg_union_exact_pct = ("union_exact_pct",  "mean"),
                     std_union_exact_pct = ("union_exact_pct",  "std"),
                 )
                 .reset_index()
    )

    order = ["E-CURRENT", "E-HIST", "DE-CURRENT", "DE-HIST"]
    records = []

    for cond in order:
        us = union_summary[union_summary["condition"] == cond]
        if us.empty:
            continue

        row: dict = {"condition": cond}
        row["n_docs"] = int(us["n_docs"].iloc[0])
        row["avg_union_exact_pct"] = round(us["avg_union_exact_pct"].iloc[0], 2)
        row["std_union_exact_pct"] = round(us["std_union_exact_pct"].iloc[0], 2)

        # Individual annotators for this condition (alphabetical)
        anns = (
            ann_summary[ann_summary["condition"] == cond]
            .sort_values("annotator_a")
        )
        for i, (_, ann_row) in enumerate(anns.iterrows(), start=1):
            row[f"ann_{i}"]            = ann_row["annotator_a"]
            row[f"exact_pct_ann_{i}"]  = round(ann_row["avg_exact_pct"], 2)

        records.append(row)

    result = pd.DataFrame(records)

    # Reorder columns for readability
    id_cols    = ["condition", "n_docs"]
    ann_cols   = sorted(
        [c for c in result.columns if c.startswith("ann_") or c.startswith("exact_pct_ann_")]
    )
    union_cols = ["avg_union_exact_pct", "std_union_exact_pct"]
    result = result[id_cols + ann_cols + union_cols]

    result.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"Written to {csv_out}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()