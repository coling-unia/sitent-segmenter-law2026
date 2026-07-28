#!/usr/bin/env python3
"""
Extracts situation-entity segments from annotation files and
produces train/dev/test split DataFrames for use in the dataset preparation step.

Usage:
    python src/prepare_data.py \\
        --xmi-dir   data/annotated_corpus/annotated_xmi \\
        --split-csv data/annotated_corpus/train_test_split.csv \\
        --output-dir data/

Outputs:
    <output-dir>/edu_df.pkl   – one row per situation-entity segment with fold labels
    <output-dir>/text_df.pkl  – one row per document (full text)
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from cassis import load_typesystem, load_cas_from_xmi

SITENT = "webanno.custom.SituationEntities"


def extract_segments_from_directory(xmi_dir, split_csv, output_dir, exclude_test_files=False):
    """
    Extract segments from XMI files and create dataframe with train/dev/test splits.

    xmi_dir: Path to directory containing XMI files
    split_csv: Path to train-test split CSV file (optional)
    output_dir: Path to directory for output dataframes as pickled files
    exclude_test_files: If True, force the four paper test documents into the test fold

    Returns a tuple (edu_df, text_df) of DataFrames
    """
    try:
        # read existing train-test split if available to reproduce previous results
        df_split = pd.read_csv(split_csv, sep="\t")
        print(f"Read train-test split from: {split_csv}")
        df_split.columns = df_split.columns.str.strip()

        dev_ratio = 0.15
        rng = np.random.default_rng(42)

        df_split = df_split.copy()

        # apply dev split per category to get dev example(s) from each category
        for category in df_split["category"].unique():
            # only split dev samples from train rows to keep test set comparable!
            mask = (df_split["category"] == category) & (df_split["fold"] == "train")
            train_indices = df_split[mask].index

            # possible remainder stays in train dev set probably smaller than 15%
            n_dev = int(np.floor(len(train_indices) * dev_ratio))

            if n_dev > 0:
                # apply dev split randomly to train rows within each category
                dev_indices = rng.choice(train_indices, size=n_dev, replace=False)
                df_split.loc[dev_indices, "fold"] = "dev"

        print("Applied dev split per category to train set, resulting fold distribution:")
        value_counts = df_split["fold"].value_counts(normalize=True)
        proportions = {index: f"{value:.2f}" for index, value in value_counts.items()}
        print("Proportions:", ", ".join([f"{k}: {v}" for k, v in proportions.items()]))

    except FileNotFoundError:
        print(f"Warning: No split .csv found, proceeding without split metadata and performing 80-10-10 split manually.")
        df_split = None

    edu_rows = []
    text_rows = []

    try:
        typesystem_path = os.path.join(xmi_dir, "TypeSystem.xml")
        with open(typesystem_path, "rb") as f:
            print(f"Loading typesystem from: {xmi_dir}")
            typesystem = load_typesystem(f)
    except FileNotFoundError:
        print(f"Warning: No typesystem provided in xmi directory.")
        typesystem = None

    xmi_files = [f for f in os.listdir(xmi_dir) if f.endswith(".xmi")]

    print(f"Found {len(xmi_files)} xmi files in {xmi_dir}\nStarting segment extraction...")

    for file in tqdm(sorted(xmi_files), desc="Processing xmi files", total=len(xmi_files)):

        file_path = os.path.join(xmi_dir, file)

        with open(file_path, "rb") as f:
            cas = load_cas_from_xmi(f, typesystem=typesystem)

        filename = file.replace(".xmi", "")

        # full text: 
        full_text = cas.sofa_string
        text_rows.append({
            "file": filename,
            "full_text": full_text
        })

        # situation entities:
        for se in cas.select(SITENT):

            try:
                primary = getattr(se, "Primary_SE_Type")
            except AttributeError:
                primary = getattr(se, "SE_Type", None)

            edu_rows.append({
                "file": filename,
                "edu_text": se.get_covered_text(),
                "begin": se.begin,
                "end": se.end,
                "primary_entity_type": primary,
                "secondary_entity_type": getattr(se, "Secondary_SE_Type", None),
                "tense_interpretation": getattr(se, "tense_interpretation", None),
                "habituality": getattr(se, "Habituality", None),
                "reported_speech": getattr(se, "reported_speech", None) == "true"
            })

    edu_df = pd.DataFrame(edu_rows)
    text_df = pd.DataFrame(text_rows)

    # add fold and category information from the train-dev-test split if available:
    if df_split is not None:
        edu_df = edu_df.merge(
            df_split[["category_filename", "fold", "category"]], 
            left_on="file", 
            right_on="category_filename", 
            how="left"
        )
        edu_df.drop(columns=["category_filename"], inplace=True)

    # no split metadata, perform an 80-10-10 split manually:
    if df_split is None:
        # shuffle edu_df rows
        edu_df = edu_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # calculate split indices
        n_total = len(edu_df)
        n_train = int(0.8 * n_total)
        n_dev = int(0.1 * n_total)

        # assign fold labels
        edu_df.loc[:n_train-1, "fold"] = "train"
        edu_df.loc[n_train:n_train+n_dev-1, "fold"] = "dev"
        edu_df.loc[n_train+n_dev:, "fold"] = "test"

        print("Applied manual 80-10-10 split on Segments.")

    print(f"Extracted {len(edu_df)} Segments, extraction complete.")

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # save pickled dataframes
    edu_output_path = os.path.join(output_dir, "edu_df.pkl")
    text_output_path = os.path.join(output_dir, "text_df.pkl")

    if exclude_test_files:
        # manual fold assignment of these four files that serve as manually segmented test examples in the paper:
        target_prefixes = (
            "email_lists-003-2183485", 
            "news_wsj_0135", 
            "travel_IntroDublin", 
            "wiki_trees"
        )

        mask = edu_df['file'].str.startswith(target_prefixes)

        # make sure to set fold to 'test'
        edu_df.loc[mask, 'fold'] = 'test'
        if mask.sum() > 0:
            print(f"Manually assigned {mask.sum()} rows from {target_prefixes} to 'test'.")

    print("Final Segment proportions:")
    row_counts = edu_df['fold'].value_counts()
    row_props = edu_df['fold'].value_counts(normalize=True)
    for fold in row_counts.index:
        print(f"{fold}: {row_counts[fold]} rows = {row_props[fold]:.2%}")

    print("Final file (document) proportions:")
    file_fold_map = edu_df.drop_duplicates(subset='file')['fold']
    file_counts = file_fold_map.value_counts()
    file_props = file_fold_map.value_counts(normalize=True)
    for fold in file_counts.index:
        print(f"{fold}: {file_counts[fold]} files = {file_props[fold]:.2%}")
    
    # save pickled dataframes to output directory for use in dataset creation script
    edu_df.to_pickle(edu_output_path)
    text_df.to_pickle(text_output_path)
    
    print(f"Saved pickled dataframes to {output_dir}")

    return edu_df, text_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract Segments from XMI files and create train/dev/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))

    parser.add_argument(
        "--xmi-dir",
        type=str,
        default=os.path.join(data_dir, "annotated_corpus/annotated_xmi"),
    )

    parser.add_argument(
        "--split-csv",
        type=str,
        default=os.path.join(data_dir, "annotated_corpus/train_test_split.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(data_dir,),
    )

    parser.add_argument(
        "--exclude-test-files",
        action="store_true",
        default=False,
        dest="exclude_test_files",
        help="Force the four paper test documents into the test fold"
    )
            
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"Starting Segment extraction with the following parameters:")
    print(f"  XMI directory: {args.xmi_dir}")
    print(f"  Split CSV: {args.split_csv}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # load edu_df for some overview statistics:
    edu_df, _ = extract_segments_from_directory(
        xmi_dir=args.xmi_dir,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        exclude_test_files=args.exclude_test_files
    )

    print(f"Segment extraction and splitting complete. Extracted {len(edu_df)} Segments.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script took {elapsed_time:.2f} seconds to run.")


if __name__ == "__main__":
    main()