#!/usr/bin/env python3
"""
Evaluates a trained model against gold-standard .tagged.tsv files and prints
per-file and aggregate segmentation metrics.

Unlike eval_experiment.py (which summarises pre-saved JSON results), this script
runs the model live: it loads best_model.pt, predicts on each file, and computes
metrics on the fly.

Sentence boundaries are handled in two ways:
  Default:   spaCy segments the flat token list into sentences directly
             (avoids the lossy text-reconstruction problem).
  --no-spacy: blank lines in the TSV are used as sentence boundaries (legacy).

Long documents that exceed the sub-token limit are handled via binary-search
chunking: the largest chunk that fits within --max-length sub-tokens is found
and processed, then the script slides forward — no content is silently dropped.

Usage:
    python src/evaluate.py \\
        --data-dir  data/Segmentation/E-CURRENT/2_Manually-Segmented \\
        --model-dir results/<experiment_dir> \\
        --lang en

    # Write per-file prediction TSVs as well:
    python src/evaluate.py \\
        --data-dir  data/Segmentation/E-CURRENT/2_Manually-Segmented \\
        --model-dir results/<experiment_dir> \\
        --output-dir data/Segmentation/E-CURRENT
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF

LABELS   = {"B-EDU": 0, "I-EDU": 1, "O": 2}
BIO_TAGS = [tag for tag, _ in sorted(LABELS.items(), key=lambda x: x[1])]
NUM_TAGS = len(BIO_TAGS)

# optional local model cache (avoids re-downloading on HPC clusters)
# set via: export XLM_HUB_ROOT="/path/to/cache"
HUB_ROOT = os.environ.get("XLM_HUB_ROOT")

_TAG_MAP = {"B": "B-EDU", "I": "I-EDU"}

_SPACY_DEFAULTS: Dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
}


def _load_spacy(model_name: str):
    """Load a spaCy model, auto-downloading it if it is not installed."""
    try:
        import spacy
        return spacy.load(model_name)
    except ImportError:
        sys.exit("spaCy is not installed.  Run:  pip install spacy")
    except OSError:
        print(f"spaCy model '{model_name}' not found — downloading...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        import spacy
        return spacy.load(model_name)


def _spacy_sentence_spans(
    words: List[str],
    nlp,
) -> List[Tuple[int, int]]:
    """
    Return (start, end) word-index spans for each sentence in words.

    Constructs a spaCy Doc directly from the token list (no text reconstruction)
    and runs only the sentence-boundary component. Falls back to a single span
    covering all words if the pipeline has no sentence boundary detector.
    """
    from spacy.tokens import Doc

    doc = Doc(nlp.vocab, words=words)

    _SENT_COMPONENTS = {"senter", "sentencizer", "parser", "trainable_senter"}
    for name, pipe in nlp.pipeline:
        if name in _SENT_COMPONENTS:
            pipe(doc)
            break

    if not doc.has_annotation("SENT_START"):
        return [(0, len(words))]

    return [(sent.start, sent.end) for sent in doc.sents]


def segment_words_with_spacy(
    words: List[str],
    tags:  List[str],
    nlp,
) -> List[Tuple[List[str], List[str]]]:
    """
    Split a flat token list into sentences using spaCy and return
    a list of (words, gold_tags) tuples — one per sentence.
    """
    if not words:
        return []

    spans = _spacy_sentence_spans(words, nlp)
    return [
        (words[s:e], tags[s:e])
        for s, e in spans
        if s < e 
    ]



def read_tsv_file_flat(path: Path) -> Tuple[List[str], List[str]]:
    """
    Read a .tsv file as a flat (words, tags) pair, ignoring blank lines.
    Tags are mapped from short form (B/I) to full BIO labels via _TAG_MAP.
    """
    words: List[str] = []
    tags:  List[str] = []

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "":
                continue                    
            parts = line.split("\t")
            if len(parts) < 2:
                continue                    
            words.append(parts[0])
            tags.append(_TAG_MAP.get(parts[1].strip(), "O"))

    return words, tags


def read_tsv_file_legacy(path: Path) -> List[Tuple[List[str], List[str]]]:
    """
    Read a .tsv file split into sentences by blank lines (legacy format).
    Returns a list of (words, gold_tags) tuples — one per blank-line block.
    """
    sentences: List[Tuple[List[str], List[str]]] = []
    words: List[str] = []
    tags:  List[str] = []

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "":
                if words:
                    sentences.append((words[:], tags[:]))
                    words.clear()
                    tags.clear()
            else:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                words.append(parts[0])
                tags.append(_TAG_MAP.get(parts[1].strip(), "O"))

    if words:
        sentences.append((words, tags))

    return sentences


def read_data_dir_flat(
    data_dir: Path,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Return {filename: (words, gold_tags)} for every *.tsv in data_dir."""
    files = sorted(data_dir.glob("*.tsv"))
    if not files:
        sys.exit(f"No .tsv files found in {data_dir}")
    return {f.name: read_tsv_file_flat(f) for f in files}


def read_data_dir_legacy(
    data_dir: Path,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Return {filename: [(words, gold_tags), ...]} for every *.tsv."""
    files = sorted(data_dir.glob("*.tsv"))
    if not files:
        sys.exit(f"No .tsv files found in {data_dir}")
    return {f.name: read_tsv_file_legacy(f) for f in files}



class XLMEduModel(torch.nn.Module):
    def __init__(self, pretrained: str):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(pretrained, output_hidden_states=False)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, NUM_TAGS)
        self.crf        = CRF(NUM_TAGS, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return self.classifier(outputs.last_hidden_state)

    def decode(
        self,
        emissions:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[int]]:
        return self.crf.decode(emissions.float(), mask=attention_mask.bool())


def resolve_model_path(model_id: str) -> str:
    """
    Return a local filesystem path to the model if found in the HUB_ROOT cache
    directory, otherwise return model_id unchanged for HuggingFace Hub download.
    """
    if not HUB_ROOT:
        return model_id
    cache_subdir  = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(HUB_ROOT, cache_subdir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_id
    snaps = [
        os.path.join(snapshots_dir, s)
        for s in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, s))
    ]
    return max(snaps, key=os.path.getmtime) if snaps else model_id


def find_run_dirs(root: str) -> List[Path]:
    """
    Return all run directories under root that contain both best_model.pt and
    config.json. If root itself is a run directory it is returned directly.
    """
    root = Path(root)
    if (root / "best_model.pt").exists() and (root / "config.json").exists():
        return [root]
    hits = []
    for candidate in root.rglob("best_model.pt"):
        run_dir = candidate.parent
        if (run_dir / "config.json").exists():
            hits.append(run_dir)
    return sorted(hits)


def load_model(
    run_dir:  Path,
    device:   torch.device,
    model_id: Optional[str] = None,
) -> Tuple[XLMEduModel, AutoTokenizer]:
    """
    Load the fine-tuned model and tokenizer from a training output directory.

    Reads config.json to determine the base model ID. If config.json records a
    resolved local path (model_resolved) and that path still exists, it is used
    directly; otherwise resolve_model_path() checks the HUB_ROOT cache and falls
    back to HuggingFace Hub. model_id overrides the config value when given.
    """
    config_path = run_dir / "config.json"
    ckpt_path   = run_dir / "best_model.pt"

    with open(config_path) as f:
        config = json.load(f)

    model_id   = model_id or config.get("model", config.get("model_slug", "FacebookAI/xlm-roberta-large"))
    saved_path = config.get("model_resolved")
    model_path = saved_path if (saved_path and os.path.isdir(saved_path)) else resolve_model_path(model_id)

    print(f"  Model id:   {model_id}")
    print(f"  Path:       {model_path}")
    print(f"  Checkpoint: {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model     = XLMEduModel(model_path).to(device)
    state     = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def _predict_words(
    words:      List[str],
    model:      XLMEduModel,
    tokenizer:  AutoTokenizer,
    device:     torch.device,
    max_length: int = 512,
) -> List[str]:
    """
    Predict BIO tags for a flat list of words, handling sequences longer than
    max_length via binary-search chunking.

    For each chunk, a binary search finds the largest prefix of remaining words
    whose sub-token encoding fits within max_length. The model predicts on that
    chunk, the first-sub-token label is mapped back to word level, then the
    window slides forward — no words are silently dropped.
    """
    if not words:
        return []

    n_words   = len(words)
    pred_tags = ["O"] * n_words
    chunk_start = 0

    while chunk_start < n_words:
        lo, hi = chunk_start + 1, n_words
        while lo < hi:
            mid = (lo + hi + 1) // 2
            enc_test = tokenizer(
                words[chunk_start:mid],
                is_split_into_words=True,
                return_tensors="pt",
                truncation=False,
                padding=False,
            )
            if enc_test["input_ids"].shape[1] <= max_length:
                lo = mid
            else:
                hi = mid - 1

        chunk_end   = lo
        chunk_words = words[chunk_start:chunk_end]

        enc = tokenizer(
            chunk_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            emissions = model(inputs)
            decoded   = model.decode(emissions, inputs["attention_mask"])[0]

        chunk_word_tags: Dict[int, str] = {}
        for wid, tag_idx in zip(enc.word_ids(), decoded):
            if wid is None:
                continue
            if wid not in chunk_word_tags:
                chunk_word_tags[wid] = BIO_TAGS[tag_idx]

        for local_wid, tag in chunk_word_tags.items():
            pred_tags[chunk_start + local_wid] = tag

        chunk_start = chunk_end

    return pred_tags


try:
    from nltk.metrics.segmentation import windowdiff
    _HAS_WINDOWDIFF = True
except ImportError:
    _HAS_WINDOWDIFF = False


def compute_metrics(
    all_preds:  List[List[str]],
    all_labels: List[List[str]],
) -> Dict[str, float]:
    flat_preds  = [tag for seq in all_preds  for tag in seq]
    flat_labels = [tag for seq in all_labels for tag in seq]

    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
        flat_labels, flat_preds,
        labels=["B-EDU", "I-EDU"], average="weighted", zero_division=0,
    )
    (b_edu_p,), (b_edu_r,), (b_edu_f1,), _ = precision_recall_fscore_support(
        flat_labels, flat_preds,
        labels=["B-EDU"], average=None, zero_division=0,
    )
    accuracy = accuracy_score(flat_labels, flat_preds)

    exact_match = sum(
        p == l for p, l in zip(all_preds, all_labels)
    ) / max(1, len(all_labels))

    # WindowDiff
    window_diff: float | str = "nan"
    if _HAS_WINDOWDIFF:
        total_boundaries = sum(t == "B-EDU" for t in flat_labels)
        avg_seg_length   = len(flat_labels) / max(1, total_boundaries)
        k                = max(1, round(avg_seg_length / 2))
        wd_scores = []
        for pred_seq, label_seq in zip(all_preds, all_labels):
            if len(label_seq) < 2 * k + 1:
                continue
            wd_scores.append(
                windowdiff(
                    [t == "B-EDU" for t in label_seq],
                    [t == "B-EDU" for t in pred_seq],
                    k=k, boundary=True,
                )
            )
        window_diff = float(np.mean(wd_scores)) if wd_scores else "nan"

    return {
        "overall_precision": float(overall_p),
        "overall_recall":    float(overall_r),
        "overall_f1":        float(overall_f1),
        "b_edu_precision":   float(b_edu_p),
        "b_edu_recall":      float(b_edu_r),
        "b_edu_f1":          float(b_edu_f1),
        "accuracy":          float(accuracy),
        "exact_match":       float(exact_match),
        "window_diff":       window_diff,
    }


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    width = 42
    if title:
        print(f"\n{'─' * width}")
        print(f"  {title}")
        print(f"{'─' * width}")
    rows = [
        ("Overall F1  (B-EDU + I-EDU weighted)", "overall_f1"),
        ("Overall P",                             "overall_precision"),
        ("Overall R",                             "overall_recall"),
        ("B-EDU  F1",                             "b_edu_f1"),
        ("B-EDU  P",                              "b_edu_precision"),
        ("B-EDU  R",                              "b_edu_recall"),
        ("Accuracy",                              "accuracy"),
        ("Exact-match (spaCy sentence level)",    "exact_match"),
        ("WindowDiff  (lower = better)",          "window_diff"),
    ]
    for label, key in rows:
        print(f"  {label:<40} {_fmt(metrics[key])}")
    print(f"{'─' * width}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate situation-entity segmentation against gold .tagged.tsv files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", 
        required=True,
        help="Directory containing *.tagged.tsv gold files.",
    )

    parser.add_argument(
        "--model-dir", 
        required=True,
        help="Run directory (best_model.pt + config.json) or parent thereof.",
    )

    parser.add_argument(
        "--run", 
        default=None,
        help="Substring to select among multiple run directories found under --model-dir.",
    )
    parser.add_argument(
        "--model-id", 
        default=None,
        help="Override the HuggingFace model ID from config.json.",
    )

    spacy_group = parser.add_mutually_exclusive_group()
    spacy_group.add_argument(
        "--spacy-model", 
        default=None, 
        metavar="MODEL_NAME",
        help="Full spaCy model name (e.g. en_core_web_sm, de_core_news_lg). "
             "Auto-downloaded if not installed. Mutually exclusive with --lang.",
    )

    spacy_group.add_argument(
        "--lang", 
        choices=list(_SPACY_DEFAULTS.keys()), 
        default=None,
        help="Language shorthand ("", ".join(f"'{k}'→{v}" for k, v in _SPACY_DEFAULTS.items()) + "). Mutually exclusive with --spacy-model.",
    )

    parser.add_argument(
        "--no-spacy", 
        action="store_true",
        help="Use blank-line boundaries from the TSV instead of spaCy (legacy mode).",
    )

    parser.add_argument(
        "--anchor", 
        default="DE-CURRENT",
        help="Directory name in --data-dir used to anchor the output path structure.",
    )

    parser.add_argument(
        "--model-dir-name", 
        default="Model_DONE",
        help="Leaf directory name under Automatically-Segmented for prediction output.",
    )

    parser.add_argument(
        "--device", 
        default=None,
        help="Device to run on, e.g. 'cpu', 'cuda', 'cuda:0' (auto-detected if not set).",
    )

    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Maximum sub-token sequence length per chunk.",
    )

    parser.add_argument(
        "--output-dir", 
        default=None,
        help="If set, write per-file prediction TSVs to this directory.",
    )
    
    args = parser.parse_args()

    nlp = None
    if not args.no_spacy:
        if args.spacy_model:
            spacy_model_name = args.spacy_model
        elif args.lang:
            spacy_model_name = _SPACY_DEFAULTS[args.lang]
        else:
            spacy_model_name = _SPACY_DEFAULTS["en"]   # sensible default
        print(f"Loading spaCy model: {spacy_model_name}")
        nlp = _load_spacy(spacy_model_name)
        print(f"spaCy pipeline:      {nlp.pipe_names}\n")
    else:
        print("[INFO] spaCy disabled — using TSV blank-line sentence boundaries.\n")

    device = (torch.device(args.device) if args.device else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    run_dirs = find_run_dirs(args.model_dir)
    if not run_dirs:
        sys.exit(f"No run directory found under {args.model_dir}")
    if args.run:
        run_dirs = [d for d in run_dirs if args.run in d.name]
        if not run_dirs:
            sys.exit(f"--run '{args.run}' matched no directory.")
    if len(run_dirs) > 1:
        print(f"Multiple runs found; using {run_dirs[0].name}  "
              f"(pass --run <substring> to select a different one)")
    run_dir = run_dirs[0]
    print(f"Run directory: {run_dir}\n")

    model, tokenizer = load_model(run_dir, device, model_id=args.model_id)
    print()

    data_dir = Path(args.data_dir).resolve()

    if args.no_spacy:
        # blank-line boundaries, sentences already split
        all_files_legacy = read_data_dir_legacy(data_dir)

        def get_sentences(fname):
            return all_files_legacy[fname]

        file_names = list(all_files_legacy.keys())
    else:
        # read each file as a flat token list, then segment
        all_files_flat = read_data_dir_flat(data_dir)

        def get_sentences(fname):
            words, tags = all_files_flat[fname]
            return segment_words_with_spacy(words, tags, nlp)

        file_names = list(all_files_flat.keys())

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        anchor     = args.anchor
        data_parts = data_dir.parts
        try:
            anchor_idx = next(i for i, p in enumerate(data_parts) if p == anchor)
            anchor_dir = data_parts[anchor_idx]
        except StopIteration:
            print(f"[WARN] Anchor '{anchor}' not found in data_dir path; "
                  f"writing files flat into {output_dir}")
            anchor_dir = "."
        file_output_dir = (
            output_dir / anchor_dir / "Automatically-Segmented" / args.model_dir_name
        )
        file_output_dir.mkdir(parents=True, exist_ok=True)

    global_preds:  List[List[str]] = []
    global_labels: List[List[str]] = []

    for fname in file_names:
        sentences = get_sentences(fname)
        if not sentences:
            print(f"[SKIP] {fname} — no sentences after segmentation")
            continue

        file_preds:  List[List[str]] = []
        file_labels: List[List[str]] = []
        pred_rows:   List[Tuple[str, str]] = []

        for words, gold_tags in sentences:
            pred_tags = _predict_words(
                words, model, tokenizer, device, max_length=args.max_length
            )

            if len(pred_tags) != len(gold_tags):
                print(
                    f"  [WARN] {fname}: pred/gold length mismatch "
                    f"({len(pred_tags)} vs {len(gold_tags)}) — sentence skipped"
                )
                continue

            file_preds.append(pred_tags)
            file_labels.append(gold_tags)
            for w, p in zip(words, pred_tags):
                pred_rows.append((w, p))

        if not file_preds:
            print(f"[SKIP] {fname} — no valid sentences")
            continue

        metrics = compute_metrics(file_preds, file_labels)
        n_sent  = len(file_preds)
        n_words = sum(len(s) for s in file_labels)
        print_metrics(
            metrics,
            title=f"{fname}  ({n_sent} sentences, {n_words} tokens)",
        )

        global_preds.extend(file_preds)
        global_labels.extend(file_labels)

        if output_dir:
            out_path = file_output_dir / fname.replace(
                ".tagged.tsv", ".tagged_automatic.tsv"
            )
            with open(out_path, "w", encoding="utf-8") as fh:
                for w, p in pred_rows:
                    short = p.replace("-EDU", "") if p != "O" else "O"
                    fh.write(f"{w}\t{short}\n")
            print(f"  Predictions written → {out_path}")

    if len(file_names) > 1 and global_preds:
        agg         = compute_metrics(global_preds, global_labels)
        total_words = sum(len(s) for s in global_labels)
        print_metrics(
            agg,
            title=f"AGGREGATE  ({len(global_preds)} sentences, {total_words} tokens)",
        )


if __name__ == "__main__":
    main()