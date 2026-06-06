"""Download the BraTS 2021 mirror hosted on Hugging Face (no auth required).

Source: ``rocky93/BraTS_segmentation`` — 1,251 cases with 4 MRI modalities
(``flair``, ``t1``, ``t1ce``, ``t2``) plus the official segmentation label
(``seg``). NIfTI layout matches what ``prepare_dataset.py`` already expects,
so once downloaded it can be plugged into the segmentation training pipeline
with no code changes.

Total size ≈ 15 GB. ``huggingface_hub`` caches downloads and supports resume
on transient failures, so re-running the script after an interruption is
safe.

Usage
-----
    # Dry run — print what would be downloaded
    uv run python scripts/download_hf_brats2021.py --dry-run

    # Download 50 cases only (useful for incremental experiments)
    uv run python scripts/download_hf_brats2021.py --max-cases 50

    # Full download (≈15 GB)
    uv run python scripts/download_hf_brats2021.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "rocky93/BraTS_segmentation"
REPO_TYPE = "dataset"
MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")


def list_cases() -> list[str]:
    info = HfApi().dataset_info(REPO_ID)
    cases = sorted(
        {
            s.rfilename.split("/")[0]
            for s in info.siblings
            if s.rfilename.startswith("BraTS2021_")
        }
    )
    return cases


def download_cases(cases: list[str], out_root: Path, max_workers: int) -> None:
    """Download all files for the given cases in parallel via snapshot_download.

    ``snapshot_download`` reuses the local cache so reruns skip already-fetched
    files. We pass ``allow_patterns`` to fetch only the cases we want when the
    caller selected a subset.
    """
    allow_patterns = [f"{case}/*.nii.gz" for case in cases]
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(out_root),
        allow_patterns=allow_patterns,
        max_workers=max_workers,
        tqdm_class=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/BraTS2021_HF"),
        help="Output root (default: datasets/BraTS2021_HF)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap on number of cases to fetch (default: all 1,251).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip this many cases at the start (useful for resuming).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8). HF allows up to ~16.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Listing cases in {REPO_ID} …")
    cases = list_cases()
    print(f"  found {len(cases)} cases on Hugging Face")

    selected = cases[args.start :]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    if not selected:
        print("nothing to download with the current --start / --max-cases combo")
        return

    print(
        f"Will download {len(selected)} case(s) "
        f"(≈ {len(selected) * 12} MB) → {args.out}"
    )
    if args.dry_run:
        for c in selected[:5]:
            print(f"  would fetch {c}/{{{','.join(MODALITIES)}}}.nii.gz")
        if len(selected) > 5:
            print(f"  … and {len(selected) - 5} more")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Starting parallel download with {args.workers} workers …")
    try:
        download_cases(selected, args.out, max_workers=args.workers)
    except Exception as exc:  # noqa: BLE001
        print(f"snapshot_download raised: {exc}", file=sys.stderr)
        sys.exit(1)

    # Sanity check: count locally-present cases that have all 5 modalities.
    complete = 0
    missing: list[str] = []
    for case in selected:
        case_dir = args.out / case
        if all((case_dir / f"{case}_{m}.nii.gz").exists() for m in MODALITIES):
            complete += 1
        else:
            missing.append(case)
    print(f"\nDone. Complete cases: {complete}/{len(selected)}")
    if missing:
        print(f"  incomplete: {missing[:10]}{'…' if len(missing) > 10 else ''}")
        sys.exit(1)


if __name__ == "__main__":
    main()
