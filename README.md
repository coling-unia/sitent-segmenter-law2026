# Situation Entity Segmenter (LAW 2026)

This repository contains data, code, and results accompanying our paper *Cross-Linguistic
Situation Entity Segmentation for Discourse Analysis in Diachronic English and German Text*,
presented at the 20th Linguistic Annotation Workshop (LAW 2026).

> Hanna Schmück, Veronika Urban, Xaver Krückl, Sonja Zeman, Claudia Claridge and Annemarie
> Friedrich. *Cross-Linguistic Situation Entity Segmentation for Discourse Analysis in
> Diachronic English and German Text*. In Proceedings of the 20th Linguistic Annotation
> Workshop (LAW 2026), San Diego, California, USA, July 2026.
> ([aclanthology.org/2026.law-main.8](https://aclanthology.org/2026.law-main.8/))

A trained version of the model is available on Hugging Face:
[coling-unia/situation-entity-segmenter](https://huggingface.co/coling-unia/situation-entity-segmenter).

## Repository structure

```
.
├── Segmenter/     # Model training/prediction pipeline, data, and results
│                  # (see Segmenter/README.md for full technical documentation)
├── Evaluation/    # Scripts for inter-annotator agreement, corpus statistics,
│                  # and model evaluation against manual annotations
├── Results/       # Error analysis results (manual annotation vs. model)
└── LICENSE.md
```

### Segmenter

[Segmenter](Segmenter/) contains the full pipeline for the segmentation model:
extracting training segments from annotated data, building tokenized datasets, fine-tuning
XLM-RoBERTa + CRF, and predicting segment boundaries on new text. See
[Segmenter/README.md](Segmenter/README.md) for setup instructions and pipeline usage.

### Evaluation

[Evaluation](Evaluation/) contains standalone scripts used to:

- convert raw text and MASC XML annotations into tagged, tokenized data for the pipeline,
- compute inter-annotator agreement (span-level and document-level) on manually segmented data,
- compute corpus statistics (token/segment counts per language variety),
- compare model predictions against gold and manual annotations, and summarise the results.

Each script documents its own expected input/output layout and usage.

### Results

[Results](Results/) contains the outputs of the manual error analysis comparing
annotator disagreements and model predictions, as far as data release allows. For the segmenter results, check the respective directory.

## Installation

```bash
cd Segmenter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Citation

If you use this segmenter or the accompanying data, please cite:

```bibtex
@inproceedings{schmuck-etal-2026-cross,
    title     = "Cross-Linguistic Situation Entity Segmentation for Discourse Analysis in Diachronic {E}nglish and {G}erman Text",
    author    = {Schm{\"u}ck, Hanna and Urban, Veronika and Kr{\"u}ckl, Xaver and Zeman, Sonja and Claridge, Claudia and Friedrich, Annemarie},
    editor    = "Liu, Yang Janet and Gessler, Luke",
    booktitle = "Proceedings of the 20th Linguistic Annotation Workshop (LAW XX)",
    month     = jul,
    year      = "2026",
    address   = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2026.law-main.8/",
    doi       = "10.18653/v1/2026.law-main.8",
    pages     = "95--112"
}
```

## License

This project's own code is licensed under the MIT License. The annotated corpus data used for training in
[Segmenter/data/annotated_corpus](Segmenter/data/annotated_corpus) is licensed separately under
the Apache License, Version 2.0. See [LICENSE.md](LICENSE.md) for full details.
