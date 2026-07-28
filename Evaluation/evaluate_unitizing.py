"""
evaluate_unitizing.py — Span Agreement Evaluator
-------------------------------------------------

Input
- Two B I tagged TSV files

Output
- A summary of inter-annotator agreement at the span level.

Usage:
    python evaluate_unitizing.py <file_a.tagged.tsv> <file_b.tagged.tsv>

"""

import sys
import difflib
from dataclasses import dataclass
from pathlib import Path



# Data structure returned by evaluate_pair()
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    name_a:           str
    name_b:           str
    spans_a:          int
    spans_b:          int
    exact:            int
    exact_pct:        float          # % of union
    partial_overlaps: int
    only_a:           int            # spans in A with no overlap in B
    only_b:           int            # spans in B with no overlap in A
    precision:        float          # treating A as reference
    recall:           float
    f1:               float
    # Detailed span data (token-index pairs) for verbose printing
    exact_spans:      list[tuple[int, int]]
    partial_pairs:    list[tuple[tuple[int,int], tuple[int,int]]]
    no_overlap_a:     list[tuple[int, int]]
    no_overlap_b:     list[tuple[int, int]]
    # Shared token list (needed to reconstruct text from spans)
    tagged:           list[tuple[str, str]]
    # Whether whitespace tokens were silently dropped to unify the texts
    ws_unified:       bool



# I/O
# ---------------------------------------------------------------------------

def load_tagged(filepath: Path) -> list[tuple[str, str]]:
    """
    Read a tab-separated token/tag file.
    Each non-empty line must be:  token <TAB> tag

    Accepted tags: B, I, O (case-insensitive).
    Each contiguous run of O-tagged tokens is treated as a single span and
    remapped to B (first token) + I (remaining tokens), so that downstream
    span extraction handles O-regions identically to explicit B/I annotations.

    Raises ValueError on malformed input (so callers can handle it).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    raw: list[tuple[str, str]] = []
    for lineno, line in enumerate(
        filepath.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"{filepath}:{lineno}: expected 'token<TAB>tag', got {line!r}"
            )
        token, tag = parts
        tag_upper = tag.upper()
        if tag_upper not in ("B", "I", "O"):
            raise ValueError(
                f"{filepath}:{lineno}: unknown tag {tag!r} (expected B, I, or O)"
            )
        raw.append((token, tag_upper))

    # Remap O-runs: first token in each run → B, subsequent tokens → I.
    tagged: list[tuple[str, str]] = []
    in_o_run = False
    for token, tag in raw:
        if tag == "O":
            tagged.append((token, "B" if not in_o_run else "I"))
            in_o_run = True
        else:
            tagged.append((token, tag))
            in_o_run = False

    return tagged



# Span extraction
# ---------------------------------------------------------------------------

def extract_spans(tagged: list[tuple[str, str]]) -> list[tuple[int, int]]:
    """Return (start, end) index pairs (end exclusive) for each span."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, (_, tag) in enumerate(tagged):
        if tag == "B":
            if start is not None:
                spans.append((start, i))
            start = i
    if start is not None:
        spans.append((start, len(tagged)))
    return spans


# Text comparison + whitespace unification
# ---------------------------------------------------------------------------

def tokens_only(tagged: list[tuple[str, str]]) -> list[str]:
    return [tok for tok, _ in tagged]


def report_text_diff(
    tokens_a: list[str], tokens_b: list[str], name_a: str, name_b: str
) -> None:
    """Print a token-level diff with positions in each file."""
    diff = list(difflib.ndiff(tokens_a, tokens_b))
    print("\n  Token-level diff  (- = only in A,  + = only in B):")
    pos_a = pos_b = 0
    for item in diff:
        code, word = item[0], item[2:]
        if code == "-":
            print(f"    pos {pos_a:>4} in {name_a}: -{word!r}")
            pos_a += 1
        elif code == "+":
            print(f"    pos {pos_b:>4} in {name_b}: +{word!r}")
            pos_b += 1
        elif code == " ":
            pos_a += 1
            pos_b += 1


def reconcile(
    tagged_a: list[tuple[str, str]],
    tagged_b: list[tuple[str, str]],
    name_a: str,
    name_b: str,
    verbose: bool = True,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], bool] | None:
    """
    Check whether the two token sequences are identical.

    Returns (tagged_a, tagged_b, ws_unified) or None if texts differ in content.
      - ws_unified: True if whitespace tokens were dropped to make them match.
    When verbose=True, prints diffs and status messages.
    """
    tokens_a = tokens_only(tagged_a)
    tokens_b = tokens_only(tagged_b)

    if tokens_a == tokens_b:
        return tagged_a, tagged_b, False

    opcodes = difflib.SequenceMatcher(None, tokens_a, tokens_b).get_opcodes()
    ws_only_diffs: list[str] = []
    real_diffs:    list[str] = []

    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            continue
        chunk_a = tokens_a[i1:i2]
        chunk_b = tokens_b[j1:j2]
        all_ws = all(t.strip() == "" for t in chunk_a + chunk_b)
        bucket = ws_only_diffs if all_ws else real_diffs
        for t in chunk_a:
            bucket.append(f"    pos {i1:>4} in {name_a}: -{t!r}")
        for t in chunk_b:
            bucket.append(f"    pos {j1:>4} in {name_b}: +{t!r}")

    if real_diffs:
        if verbose:
            print_section("TEXT MISMATCH — files do not contain the same text")
            report_text_diff(tokens_a, tokens_b, name_a, name_b)
            print("\n  Span evaluation requires identical token sequences.")
            print("  Skipping span evaluation.")
        return None

    if verbose:
        print_section("WARNING: Whitespace-only tokens differ — unifying by dropping them")
        print(f"  {len(ws_only_diffs)} whitespace token(s) removed:")
        for line in ws_only_diffs:
            print(line)

    tagged_a = [(tok, tag) for tok, tag in tagged_a if tok.strip() != ""]
    tagged_b = [(tok, tag) for tok, tag in tagged_b if tok.strip() != ""]

    if tokens_only(tagged_a) != tokens_only(tagged_b):
        if verbose:
            print("\n WARNING: Texts still differ after whitespace removal — skipping.")
        return None

    return tagged_a, tagged_b, True


# Evaluation computation
# ---------------------------------------------------------------------------

def compute_evaluation(
    tagged_a: list[tuple[str, str]],
    tagged_b: list[tuple[str, str]],
    name_a: str,
    name_b: str,
    ws_unified: bool,
) -> EvaluationResult:
    """Pure computation — no output. Returns an EvaluationResult."""
    spans_a = extract_spans(tagged_a)
    spans_b = extract_spans(tagged_b)

    set_a = set(spans_a)
    set_b = set(spans_b)

    exact  = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    def overlaps(s1: tuple[int, int], s2: tuple[int, int]) -> bool:
        return s1[0] < s2[1] and s2[0] < s1[1]

    partial_pairs:  list[tuple[tuple[int,int], tuple[int,int]]] = []
    matched_only_a: set[tuple[int,int]] = set()
    matched_only_b: set[tuple[int,int]] = set()

    for sa in sorted(only_a):
        for sb in sorted(only_b):
            if overlaps(sa, sb):
                partial_pairs.append((sa, sb))
                matched_only_a.add(sa)
                matched_only_b.add(sb)

    no_overlap_a = only_a - matched_only_a
    no_overlap_b = only_b - matched_only_b

    total_union = len(set_a | set_b)
    exact_pct = len(exact) / total_union * 100 if total_union else 0.0

    tp = len(exact)
    fp = len(only_b)
    fn = len(only_a)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return EvaluationResult(
        name_a=name_a,
        name_b=name_b,
        spans_a=len(spans_a),
        spans_b=len(spans_b),
        exact=len(exact),
        exact_pct=exact_pct,
        partial_overlaps=len(partial_pairs),
        only_a=len(no_overlap_a),
        only_b=len(no_overlap_b),
        precision=precision,
        recall=recall,
        f1=f1,
        exact_spans=sorted(exact),
        partial_pairs=partial_pairs,
        no_overlap_a=sorted(no_overlap_a),
        no_overlap_b=sorted(no_overlap_b),
        tagged=tagged_a,   # shared token list
        ws_unified=ws_unified,
    )


# Pair evaluation
# ---------------------------------------------------------------------------
def evaluate_pair(
    path_a: Path,
    path_b: Path,
    verbose: bool = False,
) -> EvaluationResult | None:
    """
    Load two tagged TSV files, reconcile their token sequences, and return
    an EvaluationResult.  Returns None if the texts differ in content.

    Set verbose=True to print diffs and section headers (mirrors standalone
    behaviour).
    """
    name_a = path_a.stem
    name_b = path_b.stem

    tagged_a = load_tagged(path_a)
    tagged_b = load_tagged(path_b)

    result = reconcile(tagged_a, tagged_b, name_a, name_b, verbose=verbose)
    if result is None:
        return None
    tagged_a, tagged_b, ws_unified = result

    return compute_evaluation(tagged_a, tagged_b, name_a, name_b, ws_unified)



# Verbose printing (used by standalone main)
# ---------------------------------------------------------------------------

def print_result(result: EvaluationResult) -> None:
    """Print a full human-readable report for an EvaluationResult."""

    def span_text(span: tuple[int, int]) -> str:
        return " ".join(tok for tok, _ in result.tagged[span[0]:span[1]])

    if result.ws_unified:
        print_section("Whitespace tokens were unified before evaluation")

    print_section("✓ Texts match — proceeding to span evaluation")

    print_section("SPAN STATISTICS")
    print(f"  {result.name_a}: {result.spans_a} spans")
    print(f"  {result.name_b}: {result.spans_b} spans")
    delta = result.spans_a - result.spans_b
    if delta > 0:
        print(f"\n  → {result.name_a} has more spans ({delta} more)")
    elif delta < 0:
        print(f"\n  → {result.name_b} has more spans ({-delta} more)")
    else:
        print("\n  → Both annotators produced the same number of spans")

    print_section("AGREEMENT SUMMARY")
    print(f"  Exact matches:                {result.exact:>5}  ({result.exact_pct:.1f}% of union)")
    print(f"  Partial overlaps (non-exact): {result.partial_overlaps:>5}")
    print(f"  Only in {result.name_a} (no overlap): {result.only_a:>4}")
    print(f"  Only in {result.name_b} (no overlap): {result.only_b:>4}")
    print(f"\n  Treating {result.name_a} as reference:")
    print(f"    Precision: {result.precision:.3f}   Recall: {result.recall:.3f}   F1: {result.f1:.3f}")

    if result.exact_spans:
        print_section("EXACT MATCHES")
        for span in result.exact_spans:
            print(f"  [{span[0]}:{span[1]}]  \"{span_text(span)}\"")

    if result.partial_pairs:
        print_section("PARTIAL OVERLAPS")
        for sa, sb in result.partial_pairs:
            print(f"\n  {result.name_a} [{sa[0]}:{sa[1]}]: \"{span_text(sa)}\"")
            print(f"  {result.name_b} [{sb[0]}:{sb[1]}]: \"{span_text(sb)}\"")

    if result.no_overlap_a:
        print_section(f"NO MATCH — only in {result.name_a}")
        for span in result.no_overlap_a:
            print(f"  [{span[0]}:{span[1]}]  \"{span_text(span)}\"")

    if result.no_overlap_b:
        print_section(f"NO MATCH — only in {result.name_b}")
        for span in result.no_overlap_b:
            print(f"  [{span[0]}:{span[1]}]  \"{span_text(span)}\"")



def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")



# Standalone main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python evaluate_unitizing.py "
            "<file_a.tagged.tsv> <file_b.tagged.tsv>"
        )

    path_a = Path(sys.argv[1])
    path_b = Path(sys.argv[2])

    print(f"Reading {path_a}…")
    tagged_a = load_tagged(path_a)
    print(f"  {len(tagged_a)} tokens")

    print(f"Reading {path_b}…")
    tagged_b = load_tagged(path_b)
    print(f"  {len(tagged_b)} tokens")

    rec = reconcile(tagged_a, tagged_b, path_a.stem, path_b.stem, verbose=True)
    if rec is None:
        print(f"\n{'='*60}\n")
        return
    tagged_a, tagged_b, ws_unified = rec

    result = compute_evaluation(tagged_a, tagged_b, path_a.stem, path_b.stem, ws_unified)
    print_result(result)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()