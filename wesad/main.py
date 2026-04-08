"""
Entry point for the WESAD dataset preprocessing pipeline.

Usage:
    python wesad/main.py

Expected layout before running:
    datasets/
    └── WESAD.zip

The script will:
1. Extract WESAD.zip into datasets/WESAD/ (skipped if already extracted).
2. Process each subject, extracting windowed features and saving a CSV.
3. Merge all per-subject CSVs into a single ``final_wesad.csv``.
"""

from config import FINAL_CSV_NAME, OUTPUT_DIR, SUBJECTS
from pipeline import extract_dataset, merge_subject_csvs, process_subject, save_subject_csv


def run_pipeline() -> None:
    """Execute the full preprocessing pipeline for all subjects."""
    extract_dataset()

    failed_subjects: list[str] = []

    for subject_id in SUBJECTS:
        try:
            df = process_subject(subject_id)
            save_subject_csv(df, subject_id)
        except Exception as exc:
            print(f"[{subject_id}] ERROR: {exc}")
            failed_subjects.append(subject_id)

    if failed_subjects:
        print(f"\nFailed subjects: {failed_subjects}")

    processed = [s for s in SUBJECTS if s not in failed_subjects]

    print("\nMerging all subject CSVs...")
    final_df = merge_subject_csvs(processed)

    final_path = OUTPUT_DIR / FINAL_CSV_NAME
    final_df.to_csv(final_path, index=False)

    print(f"\nPipeline complete.")
    print(f"  Subjects processed : {len(processed)}")
    print(f"  Total windows      : {len(final_df)}")
    print(f"  Output             : {final_path}")


if __name__ == "__main__":
    run_pipeline()
