"""
manual_to_BI_spacy.py — Segment Tagger
---------------------------------

Input:
 - A raw text file where newlines mark segment boundaries.
 - A language code for the spaCy model to use.

Output:
    - A tagged TSV file where each line contains a token and its B/I tag, separated by
        a tab. The first token of each segment gets tag 'B'; all subsequent tokens get 'I'.
    - The output file is written to the same directory as the input file, with the suffix '.tagged.tsv'

Output format (tab-separated, one token per line):
    token    B
    token    I
    token    I
    token    B
    ...

Usage:
    python manual_to_BI_spacy.py <input.txt> <language_code> [output.tsv] 
"""

import sys
import bisect
from pathlib import Path
import spacy
import argparse


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_file(filepath: Path) -> str:
    if not filepath.exists():
        sys.exit(f"Error: file not found: {filepath}")
    return filepath.read_text(encoding="utf-8")


def write_tagged(tagged: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for token, tag in tagged:
            f.write(f"{token}\t{tag}\n")
    print(f"Written: {out_path}  ({len(tagged)} tokens)")


# ---------------------------------------------------------------------------
# Tokenisation + boundary reconstruction
# ---------------------------------------------------------------------------

def pipe_split(text: str) -> str:
    """Splits all possible segments as denoted by '|||' into newlines."""
    text = text.replace("|||", "\n")
    return text

def tokenise_and_tag(raw: str, nlp) -> list[tuple[str, str]]:
    """
    Tokenise the whole document in one spaCy pass, then reconstruct
    segment boundaries from the original newline positions using
    character offsets.

    Steps
    -----
    1. Split raw text into non-empty lines; record each line's
       [start, end) character span within a single joined string.
    2. Join lines with a single space and feed to spaCy.
    3. For every spaCy token look up which line it belongs to via its
       start character offset (tok.idx).  The first token of each new
       line group gets tag 'B'; all subsequent tokens get 'I'.
    4. Whitespace-only tokens are dropped.

    If a token's start character falls in a joining space (extremely
    rare but guarded against), it is assigned to the following line
    and a warning is printed.
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return []

    # Record each line's [start, end) char span in the joined string.
    line_spans: list[tuple[int, int]] = []
    cursor = 0
    for line in lines:
        line_spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # +1 for the joining space

    joined = " ".join(lines)
    doc = nlp(joined)

    line_starts = [s for s, _ in line_spans]
    tagged: list[tuple[str, str]] = []
    prev_line_idx: int | None = None

    for tok in doc:
        # Drop whitespace-only tokens
        if tok.text.strip() == "":
            continue

        char_pos = tok.idx
        idx = max(bisect.bisect_right(line_starts, char_pos) - 1, 0)
        _, line_end = line_spans[idx]

        if char_pos > line_end:
            # Token starts in a joining space → assign to the next line
            next_idx = idx + 1
            if next_idx < len(line_spans):
                idx = next_idx
                print(
                    f"WARNING: Token {tok.text!r} at char {char_pos} starts in a "
                    f"joining space; assigned to segment {idx + 1}."
                )

        tag = "B" if idx != prev_line_idx else "I"
        tagged.append((tok.text, tag))
        prev_line_idx = idx

    return tagged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("lang", choices=["en", "de"])
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output or in_path.with_name(in_path.stem + ".tagged.tsv")

    model = "de_core_news_sm" if args.lang == "de" else "en_core_web_sm"

    try:
        nlp = spacy.load(model)
    except OSError:
        sys.exit(f"Install the spaCy model first:\n  python -m spacy download {model}")

    print("Input:", in_path)
    print("Output:", out_path)


    print(f"Reading {in_path}…")
    raw = load_file(in_path)
    raw = pipe_split(raw)
    non_empty = sum(1 for l in raw if l)
    print(f"  {non_empty} non-empty segments")

    print("Tokenising…")
    tagged = tokenise_and_tag(raw, nlp)
    print(f"  {len(tagged)} tokens")

    write_tagged(tagged, out_path)


if __name__ == "__main__":
    main()
