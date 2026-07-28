#!/usr/bin/env python3

"""
Runs inference with a fine-tuned XLM-RoBERTa + CRF model on plain-text input,
producing token-level situation-entity segmentation labels.

The model directory must contain:
    config.json     – saved by train.py (used to identify the base model)
    best_model.pt   – weights saved by train.py

Input can be a single .txt file or a directory of .txt files.
spaCy splits each text into sentences; each sentence is predicted independently.

Output is a .tsv file with one token per line: <token>\\t<tag>

Usage (single file):
    python src/predict.py --model-dir runs/xlm-roberta-large \\
                          --input-file input.txt --output-file output.tsv

Usage (directory):
    python src/predict.py --model-dir runs/xlm-roberta-large \\
                          --input-dir texts/ --output-dir predictions/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF


LABELS = {"B-EDU": 0, "I-EDU": 1, "O": 2}
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
    Predict BI(O) tags for a list of words (one sentence).

    Sub-tokenises the words, runs the encoder and CRF decoder, then maps
    predictions back to word level by taking only the first sub-token per word
    (same alignment as during training).
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



def main():
    parser = argparse.ArgumentParser(
        description="Predict situation-entity segmentation tags for plain-text input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir", 
        required=True,
        help="Training output directory containing config.json and best_model.pt.",
    )

    parser.add_argument(
        "--input-dir",
        help="Directory of .txt files to process (batch mode).",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for .tsv files (batch mode, defaults to output_tsvs/).",
    )

    parser.add_argument(
        "--input-file",
        help="Single .txt input file (single-file mode).",
    )

    parser.add_argument(
        "--output-file", 
        default="predicted_output.tsv",
        help="Output .tsv path (single-file mode).",
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
        type=int, default=512,
        help="Maximum sub-token sequence length per sentence.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.spacy_model or _SPACY_DEFAULTS[args.lang]
    nlp = load_spacy(model_name)
    run_dir = Path(args.model_dir)

    model, tokenizer = load_model(run_dir, device)

    # determine input files
    if args.input_dir:
        input_path = Path(args.input_dir)
        files_to_process = list(input_path.glob("*.txt"))
        out_dir = Path(args.output_dir) if args.output_dir else Path("output_tsvs")
        out_dir.mkdir(parents=True, exist_ok=True)
    elif args.input_file:
        files_to_process = [Path(args.input_file)]
        out_dir = None
    else:
        sys.exit("Please provide --input-dir or --input-file")

    for file_path in files_to_process:
        print(f"Processing: {file_path.name}")
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        sentences = tokenize_text(text, nlp)
        all_rows = []

        for sent in sentences:
            tags = predict_words(sent, model, tokenizer, device, args.max_length)
            for w, t in zip(sent, tags):
                short = t.replace("-EDU", "") if t != "O" else "O"
                all_rows.append((w, short))
            # all_rows.append(("", "")) # sentence break, optional to include blank line between sentences in output TSV

        # determine output path
        if out_dir:
            save_path = out_dir / (file_path.stem + ".tsv")
        else:
            save_path = Path(args.output_file)

        with open(save_path, "w", encoding="utf-8") as f:
            for w, t in all_rows:
                if w: 
                    f.write(f"{w}\t{t}\n")
        
    print(f"Prediction done! Processed {len(files_to_process)} files.")


if __name__ == "__main__":
    main()