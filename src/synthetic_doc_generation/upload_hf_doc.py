"""
upload_doc_dataset.py
=====================
Uploads the synthetic Khmer document text-line detection dataset
to the Hugging Face Hub.

What gets uploaded
------------------
1.  Parquet dataset (HF Dataset format)
      • Enables the Hub Dataset Viewer (browse images in browser)
      • Enables `load_dataset(repo_id)` in Python
      Schema per row:
          image       : PIL Image  (JPEG bytes, variable size)
          image_id    : str        e.g. "kh_doc_000042"
          split       : str        "train" | "val"
          width       : int
          height      : int
          annotations : Sequence {
              bbox   : Sequence[float, length=4]  [cx,cy,w,h] normalised YOLO
              cls_id : int     always 0
          }

2.  Raw YOLO zip  (images/ + labels/ + dataset.yaml)
      Stored at  data/yolo_raw.zip  for direct Ultralytics training.

3.  README.md  dataset card (auto-generated, Hub-compliant YAML front-matter)

Val set is OPTIONAL — pass --no-val to skip it entirely.

Usage
-----
    # minimal (token from $HF_TOKEN)
    python upload_doc_dataset.py --repo-id your-name/khmer-doc-lines

    # explicit options
    python upload_doc_dataset.py \\
        --repo-id   your-name/khmer-doc-lines \\
        --data-dir  ./datasets \\
        --token     hf_xxxx \\
        --private \\
        --no-val \\
        --no-zip

Requirements
------------
    pip install datasets huggingface_hub Pillow tqdm
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


# ============================================================================
# SECTION 1 – YOLO label parser
# ============================================================================

def parse_yolo_label(label_path: Path) -> list[dict]:
    """
    Parse a YOLO .txt annotation file.
    Each row: <cls_id> <cx> <cy> <w> <h>  (all normalised 0–1)

    Returns a list of dicts: [{cls_id, bbox:[cx,cy,w,h]}, …]
    Empty list when the file is absent or blank.
    """
    if not label_path.exists():
        return []
    annotations = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        bbox   = [float(x) for x in parts[1:]]
        annotations.append({"cls_id": cls_id, "bbox": bbox})
    return annotations


# ============================================================================
# SECTION 2 – Dataset builder
# ============================================================================

def build_hf_dataset(
    data_dir:    Path,
    include_val: bool,
) -> "datasets.DatasetDict":
    """
    Walk data_dir/images/{split}/ and pair every image with its label file.
    Returns a HF DatasetDict ready for push_to_hub().
    """
    try:
        import datasets as hf
    except ImportError:
        sys.exit("[ERROR] 'datasets' package not installed. Run: pip install datasets")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits_to_load = ["train"]
    if include_val:
        splits_to_load.append("val")

    split_data: dict[str, list[dict]] = {}

    for split in splits_to_load:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        if not img_dir.is_dir():
            print(f"[WARNING] images/{split}/ not found – skipping.")
            continue

        img_files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in image_exts
        )
        if not img_files:
            print(f"[WARNING] No images found in images/{split}/ – skipping.")
            continue

        print(f"\n[i] Loading split '{split}': {len(img_files)} images")
        records = []
        for img_path in tqdm(img_files, desc=f"  {split}", unit="img"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            anns     = parse_yolo_label(lbl_path)

            with open(img_path, "rb") as fh:
                img_bytes = fh.read()

            # Read actual image dimensions (may vary with DocConfig)
            with Image.open(io.BytesIO(img_bytes)) as im:
                w, h = im.size

            records.append({
                "image":    img_bytes,
                "image_id": img_path.stem,
                "split":    split,
                "width":    w,
                "height":   h,
                # Columnar format: dict-of-lists inside each record
                "annotations": {
                    "bbox":   [a["bbox"]   for a in anns],
                    "cls_id": [a["cls_id"] for a in anns],
                },
            })
        split_data[split] = records

    if not split_data:
        sys.exit("[ERROR] No data found. Run generate_doc_lines.py first.")

    # Build typed feature schema
    features = hf.Features({
        "image":    hf.Image(),
        "image_id": hf.Value("string"),
        "split":    hf.Value("string"),
        "width":    hf.Value("int32"),
        "height":   hf.Value("int32"),
        "annotations": hf.Sequence({
            "bbox":   hf.Sequence(hf.Value("float32"), length=4),
            "cls_id": hf.Value("int32"),
        }),
    })

    dataset_dict = {}
    for split, records in split_data.items():
        dataset_dict[split] = hf.Dataset.from_list(records, features=features)

    return hf.DatasetDict(dataset_dict)


# ============================================================================
# SECTION 3 – YOLO zip builder
# ============================================================================

def build_yolo_zip(data_dir: Path, include_val: bool) -> bytes:
    """
    Create an in-memory ZIP of the YOLO dataset directory.
    If include_val is False, images/val/ and labels/val/ are excluded.
    """
    buf  = io.BytesIO()
    root = data_dir.resolve()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            parts = rel.parts
            # Skip val directories when not wanted
            if not include_val and len(parts) >= 2 and parts[1] == "val":
                continue
            zf.write(path, rel)

    buf.seek(0)
    return buf.read()


# ============================================================================
# SECTION 4 – README / dataset card generator
# ============================================================================

README_TEMPLATE = """\
---
license: mit
task_categories:
  - object-detection
language:
  - km
tags:
  - khmer
  - cambodia
  - document
  - text-detection
  - text-line
  - synthetic
  - YOLO
size_categories:
  - {size_cat}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
{val_config}---

# Synthetic Khmer Document Text-Line Detection Dataset

Synthetic dataset for **single-class text-line detection** on Cambodian
(Khmer) official documents — press releases, ministry letters, formal memos.

Generated with a procedural Pillow-based pipeline featuring:
- **8 layout templates** (standard, letter, announcement, report, sparse,
  two-column, memo, plain)
- **12 page sizes** from A5 to A4-landscape
- **Variable margins, font sizes, line spacing, and indentation**
- **Photometric augmentations** (brightness, blur, noise, JPEG, shadow, vignette, fold/crease)

## Class

| ID | Name | Description |
|----|------|-------------|
| 0 | `text_line` | Any horizontal line of text |

## Dataset statistics

| Split | Images |
|-------|--------|
| train | {n_train} |
{val_row}

## Schema

```python
{{
    "image":    Image(),
    "image_id": Value("string"),   # e.g. "kh_doc_000042"
    "split":    Value("string"),   # "train" | "val"
    "width":    Value("int32"),
    "height":   Value("int32"),
    "annotations": Sequence({{
        "bbox":   Sequence(Value("float32"), length=4),  # [cx,cy,w,h] normalised
        "cls_id": Value("int32"),                        # always 0
    }}),
}}
```

## Load with HF Datasets

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
sample = ds["train"][0]
print(sample["image"])          # PIL Image
print(sample["annotations"])    # dict of lists
```

## Raw YOLO files

`data/yolo_raw.zip` contains the native YOLO directory layout
(`images/`, `labels/`, `dataset.yaml`) for direct Ultralytics training:

```python
from huggingface_hub import hf_hub_download
import zipfile, pathlib

zip_path = hf_hub_download(
    repo_id   = "{repo_id}",
    filename  = "data/yolo_raw.zip",
    repo_type = "dataset",
)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("./khmer_doc_yolo")

# yolo train data=khmer_doc_yolo/dataset.yaml model=yolo11n.pt
```
"""

def build_readme(
    repo_id:     str,
    n_train:     int,
    n_val:       int,
    include_val: bool,
) -> str:
    total = n_train + (n_val if include_val else 0)
    size_cat = (
        "n<1K"        if total <  1_000 else
        "1K<n<10K"    if total < 10_000 else
        "10K<n<100K"
    )
    val_config = (
        "      - split: val\n        path: data/val-*.parquet\n"
        if include_val else ""
    )
    val_row = f"| val   | {n_val} |\n" if include_val else ""

    return README_TEMPLATE.format(
        repo_id    = repo_id,
        size_cat   = size_cat,
        val_config = val_config,
        n_train    = n_train,
        val_row    = val_row,
    )


# ============================================================================
# SECTION 5 – Upload orchestrator
# ============================================================================

def upload(
    repo_id:     str,
    data_dir:    Path,
    token:       Optional[str],
    private:     bool,
    include_val: bool,
    upload_zip:  bool,
) -> None:
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        sys.exit("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")

    api = HfApi(token=token)

    # ── 1. Ensure repo exists ─────────────────────────────────────────────────
    print(f"\n[1/4] Creating / verifying repo: {repo_id}")
    api.create_repo(
        repo_id   = repo_id,
        repo_type = "dataset",
        private   = private,
        exist_ok  = True,
    )
    print(f"      https://huggingface.co/datasets/{repo_id}")

    # ── 2. Build and push Parquet dataset ─────────────────────────────────────
    print(f"\n[2/4] Building HF Dataset (val={'yes' if include_val else 'SKIPPED'}) …")
    ds = build_hf_dataset(data_dir, include_val)

    n_train = len(ds.get("train", []))
    n_val   = len(ds.get("val",   [])) if include_val else 0
    print(f"      train={n_train}  val={n_val}")

    print("\n[3/4] Pushing Parquet shards to Hub …")
    ds.push_to_hub(
        repo_id,
        token          = token,
        commit_message = "Upload Parquet dataset",
        data_dir       = "data",
    )
    print("      Parquet shards uploaded")

    # ── 3. Upload README + optional YOLO zip ─────────────────────────────────
    print("\n[4/4] Uploading README" + (" + YOLO zip" if upload_zip else "") + " …")
    operations: list[CommitOperationAdd] = []

    readme_bytes = build_readme(repo_id, n_train, n_val, include_val).encode("utf-8")
    operations.append(CommitOperationAdd(
        path_in_repo = "README.md",
        path_or_fileobj = io.BytesIO(readme_bytes),
    ))

    if upload_zip:
        print("      Building YOLO zip …")
        zip_bytes    = build_yolo_zip(data_dir, include_val)
        zip_mb       = len(zip_bytes) / 1024 / 1024
        print(f"      Archive: {zip_mb:.1f} MB")
        operations.append(CommitOperationAdd(
            path_in_repo    = "data/yolo_raw.zip",
            path_or_fileobj = io.BytesIO(zip_bytes),
        ))

    api.create_commit(
        repo_id        = repo_id,
        repo_type      = "dataset",
        operations     = operations,
        commit_message = "Add README" + (" and YOLO zip" if upload_zip else ""),
    )
    print("      Done")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
[OK] Upload complete!
     Dataset : https://huggingface.co/datasets/{repo_id}
     Train   : {n_train} images
     Val     : {n_val if include_val else "not uploaded"}
     YOLO zip: {"data/yolo_raw.zip" if upload_zip else "skipped"}

Load with:
     from datasets import load_dataset
     ds = load_dataset("{repo_id}")
""")


# ============================================================================
# SECTION 6 – CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "Upload Khmer document text-line dataset to Hugging Face Hub.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """\
Examples:
  # upload train + val, token from env
  python upload_doc_dataset.py --repo-id myname/khmer-doc-lines

  # train only, private repo, skip zip
  python upload_doc_dataset.py \\
      --repo-id  myname/khmer-doc-lines \\
      --no-val \\
      --private \\
      --no-zip

  # point at a custom output directory
  python upload_doc_dataset.py \\
      --repo-id  myname/khmer-doc-lines \\
      --data-dir ./my_datasets
""",
    )
    p.add_argument("--repo-id",  "-r", required=True,
                   help='HF Hub repo: "username/dataset-name"')
    p.add_argument("--data-dir", "-d", default="./datasets",
                   help="Root of generate_doc_lines.py output (default: ./datasets)")
    p.add_argument("--token",    "-t", default=None,
                   help="HF write token (falls back to $HF_TOKEN)")
    p.add_argument("--private",  action="store_true",
                   help="Create a private repository")
    p.add_argument("--no-val",   action="store_true",
                   help="Skip the val split entirely (upload train only)")
    p.add_argument("--no-zip",   action="store_true",
                   help="Skip uploading the raw YOLO zip archive")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        sys.exit(
            "[ERROR] No HuggingFace token found.\n"
            "  Pass --token hf_xxxx  or  export HF_TOKEN=hf_xxxx\n"
            "  Get a write token at: https://huggingface.co/settings/tokens"
        )

    data_dir = Path(args.data_dir)
    if not (data_dir / "images" / "train").is_dir():
        sys.exit(
            f"[ERROR] Train images not found at: {data_dir / 'images' / 'train'}\n"
            "  Run generate_doc_lines.py first."
        )

    upload(
        repo_id     = args.repo_id,
        data_dir    = data_dir,
        token       = token,
        private     = args.private,
        include_val = not args.no_val,
        upload_zip  = not args.no_zip,
    )


if __name__ == "__main__":
    main()