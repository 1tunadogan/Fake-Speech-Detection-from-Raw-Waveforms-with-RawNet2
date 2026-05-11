#!/usr/bin/env python
"""Download ASVspoof 2019 LA dataset from Kaggle and link to data/LA.

Usage:
    uv run python scripts/download_dataset.py

This script downloads the LA-only subset (~7.66 GB) from Kaggle and creates
a directory link so that data/LA points to the downloaded dataset.
No changes to config.yaml are needed.
"""

import os
import shutil
import subprocess
import sys
import time

KAGGLE_HANDLE = "anishsarkar22/asvpoof-2019-dataset-la"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "LA")
TOTAL_STEPS = 5


def _print_header():
    print()
    print("+" + "-" * 58 + "+")
    print("|" + " " * 12 + "ASVspoof 2019 LA Dataset Downloader" + " " * 13 + "|")
    print("+" + "-" * 58 + "+")
    print()


def _print_step(step, message):
    print(f"\n[{step}/{TOTAL_STEPS}] {message}")


def _format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def _dataset_is_ready(data_dir):
    train_dir = os.path.join(data_dir, "ASVspoof2019_LA_train", "flac")
    if not os.path.isdir(train_dir):
        return False
    for entry in os.listdir(train_dir):
        if entry.endswith(".flac"):
            return True
    return False


def _get_dataset_stats(data_dir):
    total_files = 0
    total_size = 0
    for root, _, files in os.walk(data_dir):
        total_files += len(files)
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    return total_files, total_size


def _count_flac_files(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(".flac"))


def _remove_if_exists(path):
    if not os.path.exists(path):
        return
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def _create_junction_windows(target, link):
    if os.path.exists(link):
        return
    parent = os.path.dirname(link)
    if parent:
        os.makedirs(parent, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", link, target],
        check=True,
        capture_output=True,
        text=True,
    )


def _create_symlink_posix(target, link):
    if os.path.exists(link):
        return
    parent = os.path.dirname(link)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.symlink(target, link)


def _create_link(target, link):
    target = os.path.abspath(target)
    link = os.path.abspath(link)
    if sys.platform == "win32":
        _create_junction_windows(target, link)
    else:
        _create_symlink_posix(target, link)


def _download():
    try:
        import kagglehub
    except ImportError:
        print("  [FAIL] kagglehub not installed")
        print("  -> Run: uv add kagglehub")
        sys.exit(1)

    print(f"  Dataset : {KAGGLE_HANDLE}")
    print("  Downloading... (kagglehub progress bar below)")
    print()

    start = time.time()
    try:
        cache_path = kagglehub.dataset_download(KAGGLE_HANDLE)
    except Exception as exc:
        print(f"  [FAIL] Download failed: {exc}")
        print()
        print("  Kaggle API credentials required:")
        print("  1. https://www.kaggle.com/settings -> API -> Create New Token")
        print("  2. Place kaggle.json in ~/.kaggle/ (Linux/Mac)")
        print("     or %USERPROFILE%\\.kaggle\\ (Windows)")
        print("  3. Or set env vars: KAGGLE_USERNAME + KAGGLE_KEY")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"  [OK] Download complete ({elapsed:.1f}s)")
    print(f"  -> Cache : {cache_path}")
    return cache_path


def _verify(data_dir):
    checks = {
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt": "file",
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt": "file",
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt": "file",
        "ASVspoof2019_LA_train/flac": "audio",
        "ASVspoof2019_LA_dev/flac": "audio",
        "ASVspoof2019_LA_eval/flac": "audio",
    }

    results = {}
    for rel_path, kind in checks.items():
        full = os.path.join(data_dir, rel_path)
        if kind == "file":
            results[rel_path] = os.path.isfile(full)
        else:
            results[rel_path] = _count_flac_files(full)

    print("  Structure check:")
    for rel_path, result in results.items():
        if isinstance(result, bool):
            mark = "[OK]" if result else "[FAIL]"
            print(f"    {mark} {rel_path}")
        else:
            mark = "[OK]" if result > 0 else "[FAIL]"
            print(f"    {mark} {rel_path:<50s} {result:,} files")

    return all(v if isinstance(v, bool) else v > 0 for v in results.values())


def main():
    _print_header()

    # Step 1: Check existing dataset
    _print_step(1, "Checking dataset...")
    if _dataset_is_ready(DATA_DIR):
        files, size = _get_dataset_stats(DATA_DIR)
        print(f"  [OK] Found ({files:,} files, {_format_size(size)})")
        print("  -> Dataset already ready. Nothing to do.")
        print()
        return
    print("  [FAIL] Dataset not found")

    # Step 2: Download
    _print_step(2, "Downloading from Kaggle...")
    cache_path = _download()

    # Step 3: Create link
    _print_step(3, "Creating link...")
    la_source = os.path.join(cache_path, "LA")
    if not os.path.exists(la_source):
        # Some datasets may not have a top-level LA wrapper
        la_source = cache_path

    print(f"  Source : {la_source}")
    print(f"  Target : {os.path.abspath(DATA_DIR)}")

    _remove_if_exists(DATA_DIR)
    _create_link(la_source, DATA_DIR)
    print("  [OK] Link created")

    # Step 4: Verify
    _print_step(4, "Verifying structure...")
    ok = _verify(DATA_DIR)
    if not ok:
        print()
        print("  [FAIL] Verification failed!")
        sys.exit(1)
    print("  [OK] All checks passed")

    # Step 5: Summary
    _print_step(5, "Summary")
    files, size = _get_dataset_stats(DATA_DIR)
    sep = "  " + "-" * 56
    print(sep)
    print(f"  Total files : {files:,}")
    print(f"  Total size  : {_format_size(size)}")
    print(f"  Location    : {os.path.abspath(DATA_DIR)}")
    if os.path.islink(DATA_DIR):
        print("  Type        : Symlink")
    elif sys.platform == "win32":
        print("  Type        : Junction")
    print(sep)
    print()
    print("Dataset ready! Run training with:")
    print("  uv run python -m rawnet2.train --config config.yaml")
    print()


if __name__ == "__main__":
    main()
