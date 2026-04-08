"""
Entry point for the DREAMER dataset preprocessing pipeline.

Usage:
    python dreamer/main.py

Expected layout before running:
    datasets/
    └── DREAMER.zip

The script will:
1. Extract DREAMER.zip into datasets/DREAMER/ (skipped if already extracted).
2. For each subject, extract window-level band power per video and save CSVs.
3. Aggregate per-subject CSVs.
4. Merge all subjects into a single final CSV.
5. Compute brain variation statistics per video.
"""

import time

from pipeline import (
    aggregate_all_subjects,
    aggregate_subject_bandpower,
    compute_brain_variation,
    extract_dataset,
    load_dataset,
    process_subject_videos,
)


def run_pipeline() -> None:
    """Execute the full DREAMER preprocessing pipeline."""
    extract_dataset()

    print("Loading DREAMER.mat ...")
    subjects = load_dataset()
    print(f"{len(subjects)} subjects loaded.\n")

    start = time.perf_counter()
    failed: list[int] = []

    for subject_id in range(len(subjects)):
        print(f"[S{subject_id}] Processing ...")
        try:
            process_subject_videos(subject_id, subjects)
            aggregate_subject_bandpower(subject_id)
            print(f"[S{subject_id}] Done.")
        except Exception as exc:
            print(f"[S{subject_id}] ERROR: {exc}")
            failed.append(subject_id)
        print("-" * 50)

    if failed:
        print(f"\nFailed subjects: {failed}")

    print("\nMerging all subject CSVs ...")
    final_df = aggregate_all_subjects()
    print(f"Final dataset: {len(final_df)} rows.")

    print("\nComputing brain variation per video ...")
    compute_brain_variation()

    elapsed = time.perf_counter() - start
    print(f"\nPipeline complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    run_pipeline()
