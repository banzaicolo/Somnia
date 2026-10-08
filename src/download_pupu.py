#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PoPu real-data downloader — the entry point from "simulated data" to "real data"
=============================================================================

[What is PoPu?]

A publicly released research dataset (CC0 license, free to use and commercialize):
60 volunteers × 28 poses, over 50,000 real mattress pressure maps,
split into "on-mattress + under-mattress" two layers of sensors — exactly the
same scenario as our project.

Project homepage: https://github.com/rdionisio1403/PoPu

[What does this script do?]

Download the data from GitHub and extract it. Two options:

  preview (2.5MB, validate the flow small-first):  python3 src/download_pupu.py preview
  full (82MB, for actual training):                python3 src/download_pupu.py full

Data is downloaded into the pupu_data/ folder at the project root.

[Why not auto-download the full version by default?]

Respect your bandwidth and time. First use the preview to confirm the network works,
that it can be extracted and read, and only then download the full version.
=============================================================================
"""

import os
import sys
import urllib.request
import zipfile

# The real locations of the data files in the GitHub repo (raw direct links)
BASE_URL = "https://github.com/rdionisio1403/PoPu/raw/main/"
FILES = {
    "preview": ("preview_data.zip", 2.6 * 1024 * 1024),
    "full": ("PoPu_data.zip", 82 * 1024 * 1024),
}
OUT_DIR = "pupu_data"


def download(which="preview"):
    """Download and extract the specified version of the data."""
    filename, size_hint = FILES[which]
    url = BASE_URL + filename
    zip_path = os.path.join(OUT_DIR, filename)

    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(zip_path):
        print(f"[skip download] {zip_path} already exists.")
    else:
        print(f"Downloading the {which} version (about {size_hint // 1024 // 1024 + 1} MB)...")
        print(f"  URL: {url}")
        # urllib is Python's built-in download tool, no extra install needed
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Saved to {zip_path}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT_DIR)
    print(f"✅ Done. Data is in the {OUT_DIR}/ folder.")

    # Print the extracted contents so you know what is inside
    for name in sorted(os.listdir(OUT_DIR)):
        full = os.path.join(OUT_DIR, name)
        if os.path.isdir(full):
            print(f"  [dir] {name}")
        else:
            print(f"  [file] {name}  ({os.path.getsize(full) // 1024} KB)")


if __name__ == "__main__":
    # Command-line argument: preview (default) or full
    which = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if which not in FILES:
        print(f"Argument must be one of {' or '.join(FILES)}, got {which}")
        sys.exit(1)
    download(which)
