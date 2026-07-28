# Segmenter

> See the [repository root README](../README.md) for a project overview. This document covers
> the technical setup and pipeline for training and running the segmentation model.

This directory contains training data, code and results for a situation-entity segmenter of English text using a fine-tuned XLM-RoBERTa model + CRF.

The model is fine-tuned using annotated data presented in: Annemarie Friedrich, Alexis Palmer and Manfred Pinkal. Situation entity types: automatic classification of clause-level aspect. August 2016. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL). Berlin, Germany. (https://github.com/annefried/sitent).
We took over the data as provided by the authors in their repository (see data/annotated_corpus).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline

The full pipeline consists of the following steps: get the annotated corpus, extract segments
from the given annotations, build tokenised HuggingFace datasets, train the model, and
predict/evaluate results.

### Step 0 — Get the data

This repository ships without the underlying annotated corpus. Clone the
[sitent](https://github.com/annefried/sitent) repository and copy its `annotated_corpus`
folder into `data/`:

```bash
git clone https://github.com/annefried/sitent.git /tmp/sitent
mkdir -p data
cp -r /tmp/sitent/annotated_corpus data/annotated_corpus
```

### Step 1 — Extract segments from XMI annotations

Reads the annotated XMI files, applies the train/dev/test split, and writes
`edu_df.pkl` (one row per segment / edu) and `text_df.pkl` (one row per document).

```bash
python src/prepare_data.py \
    --xmi-dir   data/annotated_corpus/annotated_xmi \
    --split-csv data/annotated_corpus/train_test_split.csv \
    --output-dir data/
```

To reproduce the exact paper split (four documents forced into the test fold for current English):

```bash
python src/prepare_data.py \
    --xmi-dir   data/annotated_corpus/annotated_xmi \
    --split-csv data/annotated_corpus/train_test_split.csv \
    --output-dir data/ \
    --exclude-test-files
```

### Step 2 — Build tokenised HuggingFace datasets

Splits each document into spaCy sentences, aligns BI(O) labels with XLM-RoBERTa
sub-tokens, and saves HuggingFace `Dataset` objects to disk.

```bash
python src/prepare_datasets.py \
    --edu-pickle  data/edu_df.pkl \
    --text-pickle data/text_df.pkl \
    --model       FacebookAI/xlm-roberta-large
```

Output is written to HuggingFace datasets under `data/prepared_datasets/xlm-roberta-large/{train,dev,test}`.

### Step 3 — Train

Fine-tunes XLM-RoBERTa + CRF on the prepared datasets with early stopping.

```bash
python src/train.py \
    --data-dir data/prepared_datasets/xlm-roberta-large \
    --model    FacebookAI/xlm-roberta-large \
    --epochs   10 \
    --batch-size 64 \
    --lr       3e-5
```

The best checkpoint is saved to `--output-dir` (defaults to a hyperparameter directory
under `results/`).
When running the script in a SLURM based hyperparameter search, results + model for each combination are saved in a directory under `results/`.

To reproduce the results in `results/`, run a hyperparameter sweep over `--lr` / `--weight-decay` /
`--seed`, giving each combination its own `--output-dir` under a shared
`results/experiment_<id>/<run_name>` directory (this layout is what `evaluate_experiment.py`
expects, see Step 5). Our own SLURM job scripts are specific to our cluster (paths, container,
partition) and not included here — you'll need to write your own array job for your cluster that
loops over the hyperparameter grid and calls `src/train.py` with the flags shown above.

### Step 4 — Predict on new text

Segments a plain-text file into situation entities using a trained model checkpoint.

```bash
python src/predict.py \
    --model-dir  results/<experiment_dir> \
    --input-file input.txt \
    --output-file output.tsv
```

Output is a tab-separated file with one token per line and its predicted BI(O) tag.
For batch processing of a directory of `.txt` files use `--input-dir` / `--output-dir` instead.

### Step 4b — Predict on pre-tokenized TSV input

`predict_tsv.py` extends the prediction script with a `--pretokenized` mode for input that
is already tokenized (e.g. manually annotated test files). Each input line is `token\thuman_tag`;
the script predicts tags and writes `token\tpredicted_tag` — human tags are not written to output.

```bash
# single file
python src/predict_tsv.py \
    --model-dir  results/<experiment_dir> \
    --input-file annotated.tsv \
    --output-file predicted.tsv \
    --pretokenized

# batch
python src/predict_tsv.py \
    --model-dir results/<experiment_dir> \
    --input-dir annotated/ --output-dir predictions/ \
    --pretokenized
```

### Step 5 — Summarise a hyperparameter experiment

`evaluate_experiment.py` reads the `results_*.json` files written during training across
all seed/hyperparameter runs in a SLURM experiment directory, groups them by hyperparameter config,
and prints a ranked table of mean ± std for each metric.

```bash
python src/evaluate_experiment.py results/experiment_XXXXX
```

To sort by other metrics (here WindowDiff) and write CSV summaries alongside the experiment directory:

```bash
python src/evaluate_experiment.py results/experiment_XXXXX \
    --sort window_diff --sort-asc --csv
```

`--csv` writes both a full `_summary.csv` and a compact `_summary_compact.csv`
(mean ± std per cell) for copy-pasting into the paper.


### 5b Evaluate against gold annotations

`evaluate.py` runs the model against gold-standard `.tagged.tsv` files and
prints per-file and aggregate metrics. Unlike `evaluate_experiment.py` (which summarises after training evaluated JSON results),
this script loads the trained model and predicts from it.

Long documents that exceed the sub-token limit are handled by binary-search chunking —
no content is silently dropped.

```bash
python src/evaluate.py \
    --data-dir  data/<path-to-gold-data-dir> \
    --model-dir results/<experiment_dir> \
    --lang en
```

To also write per-file prediction TSVs alongside the metrics, add `--output-dir`:

```bash
python src/evaluate.py \
    --data-dir  data/<path-to-gold-data-dir> \
    --model-dir results/<experiment_dir> \
    --output-dir data/<output-dir-name>
```

