#!/usr/bin/env python3
"""
Builds tokenised, label-aligned HuggingFace Datasets from the pickled DataFrames
produced by prepare_data.py, ready for training with train.py.

For each fold (train / dev / test) the script:
  1. Splits each document into sentences using spaCy.
  2. Assigns BIO labels (B-EDU, I-EDU, O) to each spaCy token using the
     character-offset annotations stored in edu_df.
  3. Sub-tokenises each sentence with the XLM-RoBERTa tokenizer and aligns
     labels to sub-tokens via word_ids() — only the first sub-token of each
     word keeps the real label; subsequent sub-tokens and special tokens
     receive -100 (ignored during training).
  4. Saves the resulting Dataset to disk in HuggingFace arrow format.

Usage:
    python src/prepare_datasets.py \\
        --edu_pickle  data/edu_df.pkl \\
        --text_pickle data/text_df.pkl \\
        --model       FacebookAI/xlm-roberta-large

Outputs:
    data/prepared_datasets/<model-slug>/{train,dev,test}/
"""

import argparse
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import spacy
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# label mapping
LABELS = {"B-EDU": 0, "I-EDU": 1, "O": 2}

DEFAULT_MODEL = "FacebookAI/xlm-roberta-large"

# optional local model cache (avoids re-downloading on HPC clusters)
# set via: export XLM_HUB_ROOT="/path/to/cache"
HUB_ROOT = os.environ.get("XLM_HUB_ROOT")


def resolve_model_path(model_id: str) -> str:
    """
    Return a local filesystem path to the model if found in the HUB_ROOT cache
    directory, otherwise return model_id unchanged so the HuggingFace Hub is used
    as a fallback.
    """
    if not HUB_ROOT:
        return model_id

    cache_subdir = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(HUB_ROOT, cache_subdir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_id

    snaps = [
        os.path.join(snapshots_dir, s)
        for s in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, s))
    ]
    if snaps:
        return max(snaps, key=os.path.getmtime)

    return model_id  # fallback: HuggingFace Hub


def _load_spacy(model_name: str = "en_core_web_sm") -> spacy.Language:
    """
    Load a spaCy pipeline by name, downloading it automatically if not installed.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Installing...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return spacy.load(model_name)


def _get_label(
    tok_idx: int,
    edu_start_chars: set,
    file_edus: List[Tuple[int, int]],
) -> int:
    """
    Assign a BIO label to a single spaCy token based on its character offset (tok_idx).

    Returns B-EDU if the token starts a new situation-entity segment, I-EDU if it
    falls inside one, and O otherwise (should not occur, safety fallback).

    tok_idx:        character offset of the token in the document (spaCy tok.idx)
    edu_start_chars: set of begin offsets of all EDUs in this document (fast O(1) lookup)
    file_edus:      list of (begin, end) character spans for all EDUs in this document
    """
    if tok_idx in edu_start_chars:
        return LABELS["B-EDU"]
    for b, e in file_edus:
        if b < tok_idx < e:
            return LABELS["I-EDU"]
    return LABELS["O"]


def build_sentence_examples(
    edu_df: pd.DataFrame,
    text_df: pd.DataFrame,
    fold: str,
    spacy_model: spacy.Language,
    tokenizer,
) -> List[Dict[str, Any]]:
    """
    Build one training example per sentence for the given fold.

    For each document in the fold, spaCy splits the full text into sentences.
    Each sentence is tokenised by the XLM-RoBERTa tokenizer and BIO labels are
    aligned to sub-tokens: the first sub-token of each word keeps its label,
    all subsequent sub-tokens and special tokens ([CLS], [SEP]) receive -100
    so they are ignored by the loss function during training.

    Returns a list of dicts with keys:
        input_ids      – sub-token IDs
        attention_mask – 1 for real tokens, 0 for padding
        labels         – aligned BIO label indices (-100 = ignored)
    """

    if fold not in edu_df["fold"].unique():
        raise ValueError(f"Fold '{fold}' not found in edu_df!")

    fold_docs = edu_df[edu_df["fold"] == fold]["file"].unique()
    examples: List[Dict[str, Any]] = []

    for file in tqdm(fold_docs, desc=f"build_sentence_examples [{fold}]"):
        row = text_df[text_df["file"] == file]["full_text"].values
        if len(row) == 0:
            continue
        full_text: str = row[0]

        mask = (edu_df["file"] == file) & (edu_df["fold"] == fold)
        file_edus: List[Tuple[int, int]] = (
            edu_df[mask][["begin", "end"]].values.tolist()
        )
        if not file_edus:
            continue

        edu_start_chars = {b for b, _ in file_edus}
        doc = spacy_model(full_text)

        for sent in doc.sents:
            s_tokens: List[str] = []
            token_labels: List[int] = []

            for tok in sent:
                label = _get_label(tok.idx, edu_start_chars, file_edus)
                s_tokens.append(tok.text)
                token_labels.append(label)

            # sub-tokenise & align labels via word_ids()
            encoding = tokenizer(
                s_tokens,
                is_split_into_words=True,
                truncation=True,
                padding=False,
                max_length=512,
            )
            if len(encoding["input_ids"]) == 512:
                print(f"WARNING: sentence truncated (file={file})")

            sub_labels: List[int] = []
            previous_word_idx = None
            for word_idx in encoding.word_ids():
                if word_idx is None:
                    sub_labels.append(-100)
                elif word_idx != previous_word_idx:
                    sub_labels.append(token_labels[word_idx])
                else:
                    sub_labels.append(-100)
                previous_word_idx = word_idx

            assert len(sub_labels) == len(encoding["input_ids"])
            examples.append({
                "input_ids":      encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels":         sub_labels,
            })

    print(f"[{fold}] {len(examples)} examples built")
    return examples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--edu-pickle",  
        default="data/edu_df.pkl"
    )

    parser.add_argument(
        "--text-pickle", 
        default="data/text_df.pkl"
    )

    parser.add_argument(
        "--out-dir",     
        default=None,
        help="Output directory for prepared datasets. Defaults to data/prepared_datasets/<model-slug> if not set."
    )

    parser.add_argument(
        "--model",       
        default=DEFAULT_MODEL,
        help="HuggingFace model ID, FacebookAI/xlm-roberta-large"
    )
    
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model_slug = args.model.split("/")[-1]

    if args.out_dir is None:
        args.out_dir = os.path.join("data", "prepared_datasets", model_slug)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Model ID:       {args.model}")
    print(f"Model slug:     {model_slug}")
    print(f"Resolved path:  {model_path}")
    print(f"Output dir:     {args.out_dir}")

    print("\nLoading spaCy model...")
    nlp = _load_spacy("en_core_web_sm")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print("Is fast tokenizer:", tokenizer.is_fast)

    edu_df  = pd.read_pickle(args.edu_pickle)
    text_df = pd.read_pickle(args.text_pickle)

    for fold in ("train", "dev", "test"):
        examples = build_sentence_examples(edu_df, text_df, fold, nlp, tokenizer)
        ds = Dataset.from_list(examples)
        path = os.path.join(args.out_dir, fold)
        ds.save_to_disk(path)
        print(f"Saved {fold} ({len(ds)} examples) -> {path}")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
