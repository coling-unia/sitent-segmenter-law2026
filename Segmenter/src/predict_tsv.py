#!/usr/bin/env python3
"""
Extends predict.py with a --pretokenized mode for TSV input.

Operates in two modes:

  Raw text mode (default):
    Reads a plain .txt file, splits into sentences with spaCy, predicts BIO tags,
    and writes token\\tB/I/O per line to a .tsv output file.

  Pre-tokenized mode (--pretokenized):
    Reads a .tsv file with token\\thuman_tag per line, reconstructs the text,
    predicts tags, and writes token\\tpredicted_tag to the output.
    The human tags are not written to output — use them for evaluation separately.

The model directory must contain config.json and best_model.pt (produced by train.py).

Usage (raw text, single file):
    python src/predict_tsv.py --model-dir runs/xlm-roberta-large \\
                              --input-file input.txt --output-file output.tsv

Usage (pre-tokenized, batch):
    python src/predict_tsv.py --model-dir runs/xlm-roberta-large \\
                              --input-dir annotated/ --output-dir predictions/ \\
                              --pretokenized
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF


LABELS   = {"B-EDU": 0, "I-EDU": 1, "O": 2}
BIO_TAGS = [tag for tag, _ in sorted(LABELS.items(), key=lambda x: x[1])]
NUM_TAGS = len(BIO_TAGS)

_SPACY_DEFAULTS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
}


class XLMEduModel(torch.nn.Module):
    def __init__(self, pretrained: str):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(pretrained)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, NUM_TAGS)
        self.crf        = CRF(NUM_TAGS, batch_first=True)

    def forward(self, batch):
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return self.classifier(outputs.last_hidden_state)

    def decode(self, emissions, attention_mask):
        return self.crf.decode(emissions.float(), mask=attention_mask.bool())


def load_model(run_dir: Path, device, model_id=None):
    """
    Load the fine-tuned model and tokenizer from a training output directory.

    Reads config.json to determine the base model ID, then loads best_model.pt
    into an XLMEduModel instance. model_id can be passed explicitly to override
    the value in config.json.
    """
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    model_id = model_id or config.get("model", "xlm-roberta-base")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = XLMEduModel(model_id).to(device)

    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, tokenizer


def load_spacy(model_name):
    import spacy
    return spacy.load(model_name)


def tokenize_text(text: str, nlp) -> List[List[str]]:
    """Split text into sentences and return a list of token lists (one per sentence)."""
    doc = nlp(text)
    return [[token.text for token in sent if token.text.strip()] for sent in doc.sents]


def predict_words(words, model, tokenizer, device, max_length=512):
    """
    Predict BIO tags for a list of words (one sentence).

    Sub-tokenises the words, runs the encoder and CRF decoder, then maps
    predictions back to word level by taking only the first sub-token per word
    (same alignment as during training). Trailing words cut off by truncation
    are assigned O.
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        emissions = model(inputs)
        decoded = model.decode(emissions, inputs["attention_mask"])[0]

    tags = []
    seen = set()

    for wid, tag_idx in zip(enc.word_ids(), decoded):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        tags.append(BIO_TAGS[tag_idx])

    # truncation guard: pad any cut-off words with O
    while len(tags) < len(words):
        tags.append("O")

    return tags


def predict_pretokenized_tsv(tsv_path: Path, model, tokenizer, nlp, device, max_length=512):
    """
    Process a pre-tokenized .tsv file where each line is: token\thuman_tag
    Returns list of tuples: (token, human_tag, predicted_tag)
    Note: human_tag is preserved internally for alignment but not written to output
    """
    # Read the TSV file
    original_tokens = []
    human_tags = []
    
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:  # Empty line (sentence boundary)
                continue
            parts = line.split('\t')
            if len(parts) >= 1 and parts[0]:  # Skip empty tokens
                token = parts[0]
                human_tag = parts[1] if len(parts) > 1 else ""
                original_tokens.append(token)
                human_tags.append(human_tag)
    
    if not original_tokens:
        print(f"Warning: No tokens found in {tsv_path}")
        return []
    
    # Reconstruct text by joining with spaces — original whitespace is lost,
    # which may affect spaCy sentence boundaries for punctuation-heavy input.
    reconstructed_text = " ".join(original_tokens)
    
    # Run through spacy to get sentence boundaries
    sentences = tokenize_text(reconstructed_text, nlp)
    
    # Predict on each sentence
    all_predicted_tokens = []
    all_predicted_tags = []
    
    for sent in sentences:
        tags = predict_words(sent, model, tokenizer, device, max_length)
        all_predicted_tokens.extend(sent)
        all_predicted_tags.extend(tags)
    
    # Align predicted tokens with original tokens
    results = []
    
    if len(all_predicted_tokens) == len(original_tokens):
        # Simple 1:1 alignment
        for i, (orig_tok, human_tag) in enumerate(zip(original_tokens, human_tags)):
            pred_tag = all_predicted_tags[i]
            short_pred = pred_tag.replace("-EDU", "") if pred_tag != "O" else "O"
            results.append((orig_tok, human_tag, short_pred))
    else:
        # Tokenization mismatch - try to align
        print(f"Warning: Token count mismatch in {tsv_path.name}")
        print(f"  Original: {len(original_tokens)}, Spacy: {len(all_predicted_tokens)}")
        print("  Attempting fuzzy alignment...")
        
        pred_idx = 0
        for orig_idx, (orig_tok, human_tag) in enumerate(zip(original_tokens, human_tags)):
            if pred_idx >= len(all_predicted_tokens):
                # No more predicted tokens, use 'O' as default
                results.append((orig_tok, human_tag, "O"))
                continue
            
            pred_tok = all_predicted_tokens[pred_idx]
            
            if orig_tok == pred_tok:
                # Exact match
                pred_tag = all_predicted_tags[pred_idx]
                short_pred = pred_tag.replace("-EDU", "") if pred_tag != "O" else "O"
                results.append((orig_tok, human_tag, short_pred))
                pred_idx += 1
            else:
                # Try to handle common tokenization differences
                # Check if original token is split in spacy output
                accumulated = pred_tok
                accumulated_tags = [all_predicted_tags[pred_idx]]
                temp_pred_idx = pred_idx + 1
                
                while accumulated != orig_tok and temp_pred_idx < len(all_predicted_tokens):
                    accumulated += all_predicted_tokens[temp_pred_idx]
                    accumulated_tags.append(all_predicted_tags[temp_pred_idx])
                    if accumulated == orig_tok:
                        # Use the first non-O tag, or 'O' if all are O
                        final_tag = "O"
                        for tag in accumulated_tags:
                            if tag != "O":
                                final_tag = tag
                                break
                        short_pred = final_tag.replace("-EDU", "") if final_tag != "O" else "O"
                        results.append((orig_tok, human_tag, short_pred))
                        pred_idx = temp_pred_idx + 1
                        break
                    temp_pred_idx += 1
                else:
                    # Couldn't align, use 'O' as default
                    results.append((orig_tok, human_tag, "O"))
                    pred_idx += 1
    
    return results


def process_raw_text(file_path: Path, model, tokenizer, nlp, device, max_length, output_path: Path):
    """Process raw text file"""
    print(f"Processing: {file_path.name}")
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    sentences = tokenize_text(text, nlp)
    all_rows = []

    for sent in sentences:
        tags = predict_words(sent, model, tokenizer, device, max_length)
        for w, t in zip(sent, tags):
            short = t.replace("-EDU", "") if t != "O" else "O"
            all_rows.append((w, short))
        all_rows.append(("", ""))  # Sentence break

    with open(output_path, "w", encoding="utf-8") as f:
        for w, t in all_rows:
            if w:
                f.write(f"{w}\t{t}\n")


def process_pretokenized_tsv(file_path: Path, model, tokenizer, nlp, device, max_length, output_path: Path):
    """Process pre-tokenized TSV file"""
    print(f"Processing: {file_path.name}")
    results = predict_pretokenized_tsv(file_path, model, tokenizer, nlp, device, max_length)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for token, human_tag, pred_tag in results:
            f.write(f"{token}\t{pred_tag}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict situation-entity BIO tags for raw text or pre-tokenized TSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir", 
        required=True,
        help="Training output directory containing config.json and best_model.pt.",
    )

    parser.add_argument(
        "--input-dir",
        help="Directory of input files to process (batch mode).",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for .tsv files (batch mode, defaults to output_predictions/).",
    )

    parser.add_argument(
        "--input-file",
        help="Single input file (single-file mode).",
    )

    parser.add_argument(
        "--output-file", 
        default="output.tsv",
        help="Output .tsv path (single-file mode).",
    )

    parser.add_argument(
        "--pretokenized", 
        action="store_true",
        help="Input is a pre-tokenized .tsv with token\\thuman_tag per line.",
    )

    parser.add_argument(
        "--lang", 
        choices=["en", "de"], 
        default="en",
        help="Language, used to select the default spaCy model.",
    )

    parser.add_argument(
        "--spacy-model", 
        default=None,
        help="spaCy model name for sentence splitting (overrides --lang default).",
    )

    parser.add_argument(
        "--device", 
        default=None,
        help="Device to run on, e.g. 'cuda' or 'cpu' (auto-detected if not set).",
    )

    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Maximum sub-token sequence length per sentence.",
    )

    args = parser.parse_args()

    # Setup Device & Model
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    model_name = args.spacy_model or _SPACY_DEFAULTS[args.lang]
    print(f"Loading spacy model: {model_name}")
    nlp = load_spacy(model_name)
    
    run_dir = Path(args.model_dir)
    print(f"Loading EDU model from: {run_dir}")
    model, tokenizer = load_model(run_dir, device)

    # Determine input files
    if args.input_dir:
        input_path = Path(args.input_dir)
        if args.pretokenized:
            files_to_process = list(input_path.glob("*.tsv"))
        else:
            files_to_process = list(input_path.glob("*.txt"))
        
        out_dir = Path(args.output_dir) if args.output_dir else Path("output_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
    elif args.input_file:
        files_to_process = [Path(args.input_file)]
        out_dir = None
    else:
        sys.exit("Error: Please provide --input-dir or --input-file")

    if not files_to_process:
        sys.exit(f"Error: No files found in {args.input_dir}")

    # Process each file
    for file_path in files_to_process:
        # Determine output path
        if out_dir:
            save_path = out_dir / (file_path.stem + ".tsv")
        else:
            save_path = Path(args.output_file)
        
        # Process based on mode
        if args.pretokenized:
            process_pretokenized_tsv(file_path, model, tokenizer, nlp, device, args.max_length, save_path)
        else:
            process_raw_text(file_path, model, tokenizer, nlp, device, args.max_length, save_path)
        
        print(f"  Saved to: {save_path}")

    print(f"\nDone! Processed {len(files_to_process)} files.")


if __name__ == "__main__":
    main()