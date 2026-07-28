"""
corpus_stats.py — Corpus statistics per language variety
---------------------------------------------------------

Input
- Recursively scans Data/Segmentation_BI for .tagged.tsv files inside *_DONE folders.
- Each file is associated with a document identity (path stripped of the annotator folder) and
    a language variety.

Output
- Data/corpus_stats.csv with one row per unique document (deduplicated across annotators
    for the same document).
- Each row contains the document's variety, token count, segment count, and derived statistics.
- At the bottom of the CSV, appended summary rows per variety with aggregate statistics.


Usage:
    python corpus_stats.py [--root <path>]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd



def annotator_from_folder(name: str) -> str:
    return name.removesuffix("_DONE")


def variety_from_top_folder(top: str) -> str:
    u = top.upper()
    lang = "DE" if u.startswith("D") else "E" if u.startswith("E") else None
    if lang is None:
        raise ValueError(f"Cannot determine language from folder '{top}'.")
    if "HIST" in u:
        return f"{lang}_HIST"
    if "CURR" in u:
        return f"{lang}_CURRENT"
    return lang


def doc_key(path: Path, segmentation_bi: Path) -> str:
    """Canonical document identity — path stripped of the annotator folder."""
    rel   = path.relative_to(segmentation_bi)
    parts = list(rel.parts)
    # parts: [variety_folder, annotator_DONE, filename]
    return "/".join([parts[0], parts[-1]])



def count_tokens(path: Path) -> int:
    """Number of non-blank lines == number of tokens."""
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def count_segments(path: Path) -> int:
    """Number of annotated segments (BIO-aware)."""
    count    = 0
    in_span  = False
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                in_span = False
                continue
            parts     = line.split("\t")
            tag       = parts[1].strip().upper() if len(parts) > 1 else "O"
            if tag == "O" or tag == "":
                in_span = False
            elif tag.startswith("I-") or tag == "I":
                if not in_span:   # recover from missing B
                    count   += 1
                    in_span  = True
            else:                 # B- or any other non-O tag
                count   += 1
                in_span  = True
    return count




def find_files(segmentation_bi: Path) -> list[tuple[Path, str, str]]:
    """Return (path, doc_id, variety) for every .tagged.tsv in a *_DONE folder."""
    results = []
    for tsv in sorted(segmentation_bi.rglob("*.tagged.tsv")):
        if not tsv.parent.name.endswith("_DONE"):
            continue
        try:
            rel = tsv.relative_to(segmentation_bi)
        except ValueError:
            continue
        variety = variety_from_top_folder(rel.parts[0])
        results.append((tsv, doc_key(tsv, segmentation_bi), variety))
    return results



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corpus statistics per language variety"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="Directory containing the Data/ folder (default: cwd)",
    )
    args = parser.parse_args()

    segmentation_bi = args.root / "Data" / "Segmentation_BI"
    out_path        = args.root / "Data" / "corpus_stats.csv"

    if not segmentation_bi.is_dir():
        sys.exit(f"Error: directory not found: {segmentation_bi}")

    files = find_files(segmentation_bi)
    if not files:
        print("No .tagged.tsv files found.")
        return

    # Deduplicate across annotators: keep the first file seen per doc_id.
    seen:     set[str]         = set()
    doc_data: dict[str, tuple] = {}

    for path, d_key, variety in files:
        if d_key in seen:
            continue
        seen.add(d_key)
        tokens   = count_tokens(path)
        segments = count_segments(path)
        doc_data[d_key] = (variety, tokens, segments)

    # Build per-file rows
    VARIETY_ORDER = ["E_HIST", "E_CURRENT", "DE_HIST", "DE_CURRENT"]
    file_rows = []
    for d_key, (variety, tokens, segments) in sorted(doc_data.items()):
        file_rows.append({
            "level":               "document",
            "variety":             variety,
            "document":            d_key,
            "n_documents":         1,
            "total_tokens":        tokens,
            "mean_tokens_per_doc": tokens,
            "min_tokens":          tokens,
            "max_tokens":          tokens,
            "total_segments":      segments,
            "mean_segs_per_doc":   segments,
            "mean_tokens_per_seg": round(tokens / segments, 2)
                                   if segments else None,
        })

    # Build per-variety summary rows
    variety_buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for d_key, (variety, tokens, segments) in doc_data.items():
        variety_buckets[variety].append((tokens, segments))

    present = [v for v in VARIETY_ORDER if v in variety_buckets]
    for v in sorted(variety_buckets):
        if v not in present:
            present.append(v)

    summary_rows = []
    for variety in present:
        data         = variety_buckets[variety]
        n_docs       = len(data)
        token_counts = [t for t, _ in data]
        seg_counts   = [s for _, s in data]
        total_tokens = sum(token_counts)
        total_segs   = sum(seg_counts)
        summary_rows.append({
            "level":               "variety_summary",
            "variety":             variety,
            "document":            "",
            "n_documents":         n_docs,
            "total_tokens":        total_tokens,
            "mean_tokens_per_doc": round(total_tokens / n_docs, 1),
            "min_tokens":          min(token_counts),
            "max_tokens":          max(token_counts),
            "total_segments":      total_segs,
            "mean_segs_per_doc":   round(total_segs / n_docs, 2),
            "mean_tokens_per_seg": round(total_tokens / total_segs, 2)
                                   if total_segs else None,
        })

    # Per-file rows first, variety summaries appended at the bottom
    df = pd.DataFrame(file_rows + summary_rows)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("\nPer-variety summary:")
    print(pd.DataFrame(summary_rows).drop(columns=["level", "document"])
            .to_string(index=False))
    print(f"\nWritten → {out_path}")


if __name__ == "__main__":
    main()