"""
batch_run_BI_tagging.py — Batch Segment Tagger
--------------------------------------

Input
  - Recursively scans Data/Segmentation
  - Includes files ending in .txt
  - By default, only folders ending with "_DONE" are treated as annotator folders
  - Language is determined by the top-level folder name: DE* → German,
    E* → English. Files in other folders are skipped with a warning.
  - Language codes can be overridden by modifying the LANGUAGE_CODES dictionary in the script.

Output

  - B I tagged .tsv files in a mirrored structure in Data/Segmentation_BI.

Usage:
    python batch_run_BI_tagging.py [--root <path>] [--dry-run]

Options:
    --root      Path to the directory containing the Data/ folder.
                Defaults to the current working directory.
    --dry-run   Print what would be run without executing anything.

"""
import argparse
import subprocess
import sys
from pathlib import Path

LANGUAGE_CODES = {"DE": "de", "E": "en"}


def find_files(segmentation_root: Path) -> list[tuple[Path, str]]:
    """
    Return (txt_path, lang) pairs for every .txt file that lives inside
    a folder whose name ends with _DONE, where lang is 'de' or 'en'
    based on the name of the folder immediately below segmentation_root. Adapt
    """
    results: list[tuple[Path, str]] = []
    skipped: list[str] = []

    for txt_path in sorted(segmentation_root.rglob("*.txt")):
        # The folder immediately containing the file must end with _DONE
        if not txt_path.parent.name.endswith("_DONE"):
            continue

        # The folder directly under segmentation_root determines the language
        try:
            relative = txt_path.relative_to(segmentation_root)
        except ValueError:
            continue

        top_folder = relative.parts[0]

        for prefix in LANGUAGE_CODES:
            if top_folder.upper().startswith(prefix):
                lang = LANGUAGE_CODES[prefix]
                break
        else:
            skipped.append(
                f"  Skipping {txt_path} — top folder '{top_folder}' "
                f"is not recognized as a language folder (as defined in LANGUAGE_CODES)"
            )
            continue   


        results.append((txt_path, lang))

    for msg in skipped:
        print(msg, file=sys.stderr)

    return results


def mirror_path(
    txt_path: Path,
    segmentation_root: Path,
    segmentation_bi_root: Path,
) -> Path:
    """
    Compute the output .tagged.tsv path by replacing the segmentation
    root with the segmentation_bi root and swapping the extension.
    """
    relative = txt_path.relative_to(segmentation_root)
    return segmentation_bi_root / relative.with_suffix(".tagged.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run manual_to_BI_spacy.py")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the Data/ folder (default: cwd)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    data_root          = args.root / "Data"
    segmentation_root  = data_root / "Segmentation"
    segmentation_bi    = data_root / "Segmentation_BI"
    tagger             = Path(__file__).parent / "manual_to_BI_spacy.py"

    if not segmentation_root.is_dir():
        sys.exit(f"Error: directory not found: {segmentation_root}")
    if not tagger.is_file():
        sys.exit(f"Error: manual_to_BI_spacy.py not found at {tagger}")

    files = find_files(segmentation_root)

    if not files:
        print("No eligible .txt files found.")
        return

    print(f"Found {len(files)} file(s) to tag.\n")

    for i, (txt_path, lang) in enumerate(files, 1):
        out_path = mirror_path(txt_path, segmentation_root, segmentation_bi)

        cmd = [sys.executable, str(tagger), str(txt_path), lang, str(out_path)]

        print(f"[{i}/{len(files)}] {txt_path.relative_to(args.root)}")
        print(f"         lang={lang}  →  {out_path.relative_to(args.root)}")

        if args.dry_run:
            print(f"         (dry run — not executed)\n")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING manual_to_BI_spacy.py exited with code {result.returncode}", file=sys.stderr)

        print()

    if args.dry_run:
        print("Dry run complete — nothing was executed.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()