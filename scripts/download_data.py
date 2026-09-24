"""Download the Home Credit Default Risk dataset.

Usage:
    python scripts/download_data.py [--dest data]

Primary source is the Kaggle competition (free account required):
https://www.kaggle.com/c/home-credit-default-risk/data
Falls back to a public Hugging Face mirror of the same file.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SOURCES = [
    # Public mirror of application_train.csv (no auth required)
    "https://huggingface.co/datasets/minhSpaceX/home_credit/resolve/main/application_train.csv",
]
EXPECTED_FILE = "application_train.csv"
EXPECTED_SIZE_BYTES = 166_133_370


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / EXPECTED_FILE

    for url in SOURCES:
        print(f"Downloading {EXPECTED_FILE} (~158 MB) from {url} ...")
        try:
            urllib.request.urlretrieve(url, target)
            size = target.stat().st_size
            if size < EXPECTED_SIZE_BYTES * 0.9:
                print(f"Downloaded file looks truncated ({size} bytes); trying next source.")
                continue
            print(f"Done: {target.resolve()} ({size:,} bytes)")
            return
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f"Source failed ({exc}).")

    print(
        "All automatic sources failed. Download application_train.csv manually from "
        "https://www.kaggle.com/c/home-credit-default-risk/data "
        f"and place it in {dest.resolve()}."
    )
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data", help="Destination directory (default: data/)")
    args = parser.parse_args()
    download(Path(args.dest))
