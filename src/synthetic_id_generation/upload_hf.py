"""
upload_to_huggingface.py
========================
Uploads the synthetic Khmer National ID detection dataset to the Hugging Face Hub.

What gets uploaded
------------------
1. A structured HF Dataset (Parquet) – enables the Hub Dataset Viewer and
   lets anyone do `load_dataset("your-username/your-dataset")`.
   Schema per row:
       image       : PIL Image (JPEG bytes embedded in Parquet)
       image_id    : str  – stem filename, e.g. "kh_id_000042"
       split       : str  – "train" | "val"
       width       : int  – always 1000
       height      : int  – always 630
       objects     : {
           bbox          : list[list[float]]  – [[x_min,y_min,w,h], …] in pixels (COCO format)
           category      : list[int]          – YOLO class id (0-11)
           category_name : list[str]          – human-readable label
       }

2. A ZIP of the raw YOLO files (images + labels + dataset.yaml) under
   data/yolo_raw.zip – for users who want to train directly with Ultralytics.

3. A README.md dataset card with YAML front-matter recognised by the Hub.

Usage
-----
    python upload_to_huggingface.py \\
        --repo-id  your-username/khmer-id-detection \\
        --dataset-dir ./output \\
        --token hf_xxxxxxxxxxxx          # or set HF_TOKEN env var
        [--private]                      # make repo private
        [--no-zip]                       # skip YOLO zip upload

Requirements
------------
    pip install datasets huggingface_hub Pillow tqdm
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Class definitions (must match generate_synthetic_id.py)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "id_number",       # 0
    "name_kh",         # 1
    "name_en",         # 2
    "dob_sex_height",  # 3
    "pob",             # 4
    "address_1",       # 5
    "address_2",       # 6
    "validity",        # 7
    "features",        # 8
    "mrz_1",           # 9
    "mrz_2",           # 10
    "mrz_3",           # 11
]

CARD_W = 1000
CARD_H = 630


# ===========================================================================
# SECTION 1 – Annotation parsing
# ===========================================================================

def yolo_to_coco_bbox(
    cx: float, cy: float, w: float, h: float,
    img_w: int = CARD_W, img_h: int = CARD_H,
) -> list[float]:
    """
    Convert normalised YOLO (cx, cy, w, h) → absolute COCO pixel
    (x_min, y_min, width, height).  Values are rounded to 2 decimals.
    """
    abs_w = w  * img_w
    abs_h = h  * img_h
    x_min = (cx * img_w) - abs_w / 2
    y_min = (cy * img_h) - abs_h / 2
    return [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)]


def parse_yolo_label(label_path: Path) -> dict:
    """
    Parse a YOLO .txt annotation file into a dict ready for the HF schema:
        {
            "bbox":          [[x_min, y_min, w, h], …],   # COCO pixels
            "category":      [int, …],
            "category_name": [str, …],
        }
    """
    bboxes: list[list[float]] = []
    categories: list[int] = []
    category_names: list[str] = []

    if not label_path.exists():
        # Image with no annotations (valid YOLO convention)
        return {"bbox": bboxes, "category": categories, "category_name": category_names}

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            bboxes.append(yolo_to_coco_bbox(cx, cy, w, h))
            categories.append(cls_id)
            category_names.append(CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown")

    return {"bbox": bboxes, "category": categories, "category_name": category_names}


# ===========================================================================
# SECTION 2 – Dataset builder
# ===========================================================================

def build_hf_dataset(dataset_dir: Path) -> "datasets.DatasetDict":
    """
    Walk the output directory produced by generate_synthetic_id.py and
    return a HF DatasetDict with 'train' and 'val' splits.

    Expected layout:
        dataset_dir/
            images/
                train/  *.jpg
                val/    *.jpg
            labels/
                train/  *.txt
                val/    *.txt
    """
    # Lazy import so the script errors early if the package is missing
    try:
        import datasets as hf_datasets
    except ImportError:
        sys.exit("[ERROR] 'datasets' package not found. Run: pip install datasets")

    split_data: dict[str, list[dict]] = {"train": [], "val": []}

    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.is_dir():
            print(f"[WARNING] images/{split}/ not found – skipping split.")
            continue

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"[i] {split}: found {len(img_files)} images")

        for img_path in tqdm(img_files, desc=f"  loading {split}", unit="img"):
            label_path = lbl_dir / (img_path.stem + ".txt")
            objects    = parse_yolo_label(label_path)

            # Read image bytes (Parquet will store the raw bytes; Hub viewer
            # will decode them automatically via the Image() feature)
            with open(img_path, "rb") as fh:
                img_bytes = fh.read()

            split_data[split].append({
                "image":      img_bytes,
                "image_id":   img_path.stem,
                "split":      split,
                "width":      CARD_W,
                "height":     CARD_H,
                "objects":    objects,
            })

    if not any(split_data.values()):
        sys.exit("[ERROR] No images found. Run generate_synthetic_id.py first.")

    # ------------------------------------------------------------------
    # Build Dataset objects with explicit feature schema
    # ------------------------------------------------------------------
    features = hf_datasets.Features({
        "image":   hf_datasets.Image(),
        "image_id": hf_datasets.Value("string"),
        "split":   hf_datasets.Value("string"),
        "width":   hf_datasets.Value("int32"),
        "height":  hf_datasets.Value("int32"),
        "objects": hf_datasets.Sequence({
            "bbox":          hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
            "category":      hf_datasets.Value("int32"),
            "category_name": hf_datasets.Value("string"),
        }),
    })

    # Reformat objects: list-of-dicts → dict-of-lists (columnar, HF convention)
    def to_columnar(records: list[dict]) -> list[dict]:
        out = []
        for rec in records:
            obj = rec["objects"]
            out.append({
                **{k: v for k, v in rec.items() if k != "objects"},
                "objects": {
                    "bbox":          obj["bbox"],
                    "category":      obj["category"],
                    "category_name": obj["category_name"],
                },
            })
        return out

    dataset_dict: dict[str, "hf_datasets.Dataset"] = {}
    for split, records in split_data.items():
        if not records:
            continue
        dataset_dict[split] = hf_datasets.Dataset.from_list(
            to_columnar(records),
            features=features,
        )

    return hf_datasets.DatasetDict(dataset_dict)


# ===========================================================================
# SECTION 3 – YOLO zip builder
# ===========================================================================

def build_yolo_zip(dataset_dir: Path) -> bytes:
    """
    Create an in-memory ZIP archive of the entire YOLO output directory
    (images/, labels/, dataset.yaml) and return the raw bytes.
    """
    buf = io.BytesIO()
    root = dataset_dir.resolve()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(root)
                zf.write(path, arcname)

    buf.seek(0)
    return buf.read()


# ===========================================================================
# SECTION 4 – README / dataset card
# ===========================================================================

README_TEMPLATE = """\
---
license: mit
task_categories:
  - object-detection
language:
  - km
  - en
tags:
  - khmer
  - cambodia
  - id-card
  - synthetic
  - YOLO
  - detection
size_categories:
  - {size_category}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: val
        path: data/val-*.parquet
---

# Synthetic Khmer National ID Card — Detection Dataset

Synthetic dataset for **multi-class text field detection** on Cambodian (Khmer)
National ID cards, generated with a procedural Pillow-based pipeline.

## Classes (12 total)

| ID | Name | Description |
|----|------|-------------|
| 0 | `id_number` | ID number (top-right) |
| 1 | `name_kh` | Full name in Khmer script |
| 2 | `name_en` | Full name in Latin/English |
| 3 | `dob_sex_height` | Date of birth, sex, height |
| 4 | `pob` | Place of birth |
| 5 | `address_1` | Address line 1 |
| 6 | `address_2` | Address line 2 |
| 7 | `validity` | Issue / expiry dates |
| 8 | `features` | Distinguishing features |
| 9 | `mrz_1` | MRZ line 1 |
| 10 | `mrz_2` | MRZ line 2 |
| 11 | `mrz_3` | MRZ line 3 |

## Dataset stats

| Split | Images |
|-------|--------|
| train | {n_train} |
| val   | {n_val} |

## Schema

```python
{{
    "image":    Image(),            # PIL JPEG
    "image_id": Value("string"),    # e.g. "kh_id_000042"
    "split":    Value("string"),    # "train" | "val"
    "width":    Value("int32"),     # 1000
    "height":   Value("int32"),     # 630
    "objects":  Sequence({{
        "bbox":          Sequence(Value("float32"), length=4),  # [x_min, y_min, w, h] pixels
        "category":      Value("int32"),
        "category_name": Value("string"),
    }}),
}}
```

## Load with 🤗 Datasets

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
```

## Raw YOLO files

The file `data/yolo_raw.zip` contains the original YOLO-format annotations
(`images/`, `labels/`, `dataset.yaml`) for direct use with Ultralytics YOLO:

```python
from huggingface_hub import hf_hub_download
import zipfile, pathlib

zip_path = hf_hub_download(repo_id="{repo_id}", filename="data/yolo_raw.zip", repo_type="dataset")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("./khmer_id_yolo")
# then: yolo train data=khmer_id_yolo/dataset.yaml model=yolo11n.pt
```

## Generation

Images were generated with a procedural Python pipeline using Pillow
for text rendering and Albumentations for photometric augmentation.
No real ID card data was used.
"""


def build_readme(repo_id: str, n_train: int, n_val: int) -> str:
    total = n_train + n_val
    if total < 1_000:
        size_cat = "n<1K"
    elif total < 10_000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    return README_TEMPLATE.format(
        repo_id=repo_id,
        n_train=n_train,
        n_val=n_val,
        size_category=size_cat,
    )


# ===========================================================================
# SECTION 5 – Upload orchestrator
# ===========================================================================

def upload(
    repo_id:     str,
    dataset_dir: Path,
    token:       Optional[str],
    private:     bool,
    upload_zip:  bool,
) -> None:
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        import datasets as hf_datasets
    except ImportError:
        sys.exit(
            "[ERROR] Required packages missing.\n"
            "Run: pip install datasets huggingface_hub"
        )

    api = HfApi(token=token)

    # ---- 1. Ensure the repo exists ----------------------------------------
    print(f"\n[1/4] Creating / verifying dataset repo: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    print(f"      ✓ https://huggingface.co/datasets/{repo_id}")

    # ---- 2. Build HF Dataset and push (Parquet) ----------------------------
    print("\n[2/4] Building HF Dataset from local files …")
    dataset_dict = build_hf_dataset(dataset_dir)

    n_train = len(dataset_dict.get("train", []))
    n_val   = len(dataset_dict.get("val",   []))
    print(f"      train={n_train}  val={n_val}")

    print("\n[3/4] Pushing Parquet dataset to Hub …")
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        commit_message="Upload Parquet dataset (images + YOLO annotations)",
        data_dir="data",          # store shards under data/
    )
    print("      ✓ Parquet shards uploaded")

    # ---- 3. Upload YOLO zip + README in one commit -------------------------
    print("\n[4/4] Uploading README and raw YOLO zip …")
    operations: list[CommitOperationAdd] = []

    # README.md
    readme_bytes = build_readme(repo_id, n_train, n_val).encode("utf-8")
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=io.BytesIO(readme_bytes),
        )
    )

    # YOLO zip
    if upload_zip:
        print("      Building YOLO zip archive …")
        zip_bytes = build_yolo_zip(dataset_dir)
        zip_size_mb = len(zip_bytes) / 1024 / 1024
        print(f"      Archive size: {zip_size_mb:.1f} MB")
        operations.append(
            CommitOperationAdd(
                path_in_repo="data/yolo_raw.zip",
                path_or_fileobj=io.BytesIO(zip_bytes),
            )
        )

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Add README and raw YOLO zip",
    )
    print("      ✓ README and YOLO zip uploaded")

    # ---- Done ---------------------------------------------------------------
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Upload complete!                                            ║
║                                                              ║
║  Dataset:  https://huggingface.co/datasets/{repo_id:<20}  ║
║                                                              ║
║  Load with:                                                  ║
║    from datasets import load_dataset                         ║
║    ds = load_dataset("{repo_id}")                            ║
╚══════════════════════════════════════════════════════════════╝
""")


# ===========================================================================
# SECTION 6 – CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload synthetic Khmer ID dataset to Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # basic upload (token from env var HF_TOKEN)
  python upload_to_huggingface.py --repo-id myname/khmer-id-detect

  # explicit token, private repo, skip zip
  python upload_to_huggingface.py \\
      --repo-id myorg/khmer-id-detect \\
      --token hf_xxxx \\
      --private \\
      --no-zip
""",
    )
    parser.add_argument(
        "--repo-id", "-r",
        required=True,
        help='HF Hub repo in the form "username/dataset-name".',
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        default="./output",
        help="Root of the generate_synthetic_id.py output (default: ./output).",
    )
    parser.add_argument(
        "--token", "-t",
        default=None,
        help="HF write token. Falls back to HF_TOKEN env var if omitted.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private repository.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        default=False,
        help="Skip uploading the raw YOLO zip archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "[ERROR] No HuggingFace token provided.\n"
            "Pass --token hf_xxxx or set the HF_TOKEN environment variable.\n"
            "Get a write token at: https://huggingface.co/settings/tokens"
        )

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        sys.exit(
            f"[ERROR] Dataset directory not found: {dataset_dir}\n"
            "Run generate_synthetic_id.py first."
        )

    upload(
        repo_id=args.repo_id,
        dataset_dir=dataset_dir,
        token=token,
        private=args.private,
        upload_zip=not args.no_zip,
    )


if __name__ == "__main__":
    main()