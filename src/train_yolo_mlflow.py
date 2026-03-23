"""
train_yolo_mlflow.py
====================
Trains multiple YOLO model variants on the synthetic Khmer National ID dataset
and tracks every run with MLflow.

How it works
------------
1.  Reads a sweep config (list of dicts) – each entry defines one training run:
    model backbone, image size, epochs, batch size, and any extra YOLO kwargs.
2.  For each run:
    a.  Sets MLFLOW_EXPERIMENT_NAME and MLFLOW_RUN env vars so Ultralytics'
        built-in MLflow callback groups all runs under the same experiment.
    b.  Enables the Ultralytics MLflow callback via `settings.update`.
    c.  Trains with `model.train(...)`.
    d.  After training ends (the built-in callback has already logged per-epoch
        metrics and artifacts), opens an *extra* MLflow run to log:
        - Final summary metrics (mAP50, mAP50-95, precision, recall, fitness)
        - Per-class AP from the val results CSV
        - Training hyp params
        - The best.pt weights as a registered model artifact
3.  After the entire sweep, queries MLflow to build a leaderboard table, prints
    it to stdout, and logs it as a CSV artifact on a dedicated "leaderboard" run.

Usage
-----
    # minimal – uses defaults
    python train_yolo_mlflow.py --dataset-dir ./output

    # full sweep with GPU
    python train_yolo_mlflow.py \\
        --dataset-dir ./output \\
        --experiment  khmer-id-detection \\
        --mlflow-uri  ./mlruns \\
        --device      0

    # then open the UI
    mlflow ui --backend-store-uri ./mlruns --port 5000

Requirements
------------
    pip install ultralytics mlflow pandas tabulate
"""

import argparse
import csv
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.pytorch
import pandas as pd
from ultralytics import YOLO, settings as ul_settings

# ---------------------------------------------------------------------------
# Class list (must match generate_synthetic_id.py)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "id_number", "name_kh", "name_en", "dob_sex_height",
    "pob", "address_1", "address_2", "validity",
    "features", "mrz_1", "mrz_2", "mrz_3",
]
NUM_CLASSES = len(CLASS_NAMES)   # 12

# ---------------------------------------------------------------------------
# Keys extracted from Ultralytics results and forwarded to MLflow
# ---------------------------------------------------------------------------
SUMMARY_METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
]


# ===========================================================================
# SECTION 1 – Sweep configuration
# ===========================================================================

def default_sweep() -> list[dict[str, Any]]:
    """
    Return the default multi-model sweep.  Edit this list freely.

    Keys
    ----
    name        : unique run name (used for MLflow run + save_dir)
    model       : Ultralytics model string – pretrained weights are downloaded
                  automatically on first use.
    epochs      : training epochs
    imgsz       : input resolution (square)
    batch       : batch size (-1 = auto)
    extra       : any additional keyword args forwarded verbatim to model.train()
    """
    return [
        # ── Nano (fastest, smallest) ────────────────────────────────────────
        dict(name="yolo11n", model="yolo11n.pt",  epochs=50, imgsz=640, batch=16, extra={}),
        # ── Small ───────────────────────────────────────────────────────────
        dict(name="yolo11s", model="yolo11s.pt",  epochs=50, imgsz=640, batch=16, extra={}),
        # ── Medium ──────────────────────────────────────────────────────────
        dict(name="yolo11m", model="yolo11m.pt",  epochs=50, imgsz=640, batch=8,  extra={}),
        # ── Nano at higher resolution ────────────────────────────────────────
        dict(name="yolo11n_1280", model="yolo11n.pt", epochs=50, imgsz=1280, batch=8,  extra={}),
        # ── YOLOv8 nano for cross-generation comparison ──────────────────────
        dict(name="yolov8n",  model="yolov8n.pt",  epochs=50, imgsz=640, batch=16, extra={}),
        # ── YOLOv8 small ────────────────────────────────────────────────────
        dict(name="yolov8s",  model="yolov8s.pt",  epochs=50, imgsz=640, batch=16, extra={}),
    ]


# ===========================================================================
# SECTION 2 – Dataset YAML helper
# ===========================================================================

def resolve_dataset_yaml(dataset_dir: Path) -> Path:
    """
    Find or generate dataset.yaml inside *dataset_dir*.
    Raises SystemExit if the images/ folder structure is missing.
    """
    yaml_path = dataset_dir / "dataset.yaml"

    # Check directory structure
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        if not img_dir.is_dir():
            sys.exit(
                f"[ERROR] Expected directory not found: {img_dir}\n"
                "Run generate_synthetic_id.py first."
            )

    if yaml_path.exists():
        return yaml_path

    # Auto-generate a minimal yaml if it is missing
    print("[WARNING] dataset.yaml not found – generating a minimal one.")
    content = (
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        f"\nnc: {NUM_CLASSES}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(content)
    print(f"[i] Written: {yaml_path}")
    return yaml_path


# ===========================================================================
# SECTION 3 – Results extraction helpers
# ===========================================================================

def extract_results_dict(results) -> dict[str, float]:
    """
    Pull final metric values out of an Ultralytics Results object and return
    a flat {metric_name: float} dict.  Keys follow the SUMMARY_METRIC_KEYS
    convention; missing keys are silently skipped.
    """
    out: dict[str, float] = {}

    # results.results_dict – available in Ultralytics ≥ 8.0
    if hasattr(results, "results_dict") and results.results_dict:
        raw = results.results_dict
        for key in SUMMARY_METRIC_KEYS:
            if key in raw:
                try:
                    out[key] = float(raw[key])
                except (TypeError, ValueError):
                    pass

    # fitness scalar
    if hasattr(results, "fitness") and results.fitness is not None:
        out.setdefault("fitness", float(results.fitness))

    return out


def extract_per_class_ap(save_dir: Path) -> dict[str, float]:
    """
    Parse the per-class AP from Ultralytics' generated *results.csv* or
    *val_predictions.json* if available.  Returns {class_name: ap50} dict.
    Falls back to an empty dict if no file is found.
    """
    # Ultralytics writes class-level AP to a predictions JSON during val
    pred_json = save_dir / "predictions.json"
    if pred_json.exists():
        try:
            data = json.loads(pred_json.read_text())
            # structure: list of {category_id, score, ...}  – not per-class AP
            pass  # JSON is per-image, not what we need here
        except Exception:
            pass

    # The most reliable source is the CSV summary written at epoch end
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        return {}

    try:
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        # Last row = final epoch
        last = df.iloc[-1].to_dict()
        return {k: float(v) for k, v in last.items() if "map" in k.lower() or "ap" in k.lower()}
    except Exception:
        return {}


def parse_hyp(save_dir: Path) -> dict[str, Any]:
    """Read hyp.yaml / args.yaml from the Ultralytics save_dir."""
    for fname in ("args.yaml", "hyp.yaml"):
        p = save_dir / fname
        if p.exists():
            try:
                import yaml
                with open(p) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


# ===========================================================================
# SECTION 4 – Single-run trainer
# ===========================================================================

def train_one(
    cfg:         dict[str, Any],
    dataset_yaml: Path,
    runs_dir:    Path,
    device:      str,
    experiment:  str,
    mlflow_uri:  str,
    workers:     int,
    cache:       bool,
) -> dict[str, Any]:
    """
    Execute a single training run according to *cfg* and return a summary dict.

    MLflow strategy
    ---------------
    Ultralytics has a *built-in* MLflow callback (enabled via settings) that:
      - starts its own run on pretrain routine end
      - logs per-epoch metrics with on_fit_epoch_end
      - logs artifacts (best.pt, last.pt, plots) on train_end

    We let Ultralytics own the per-epoch logging.  After training is done we
    open a *separate* MLflow run (tagged as the "summary" run) to log:
      - final scalar summary metrics
      - per-class AP dict
      - the best.pt weight file
      - the full results.csv
      - a link back to the Ultralytics run via a tag
    This avoids the known conflict of nesting start_run() inside the callback.
    """
    run_name   = cfg["name"]
    model_id   = cfg["model"]
    epochs     = cfg["epochs"]
    imgsz      = cfg["imgsz"]
    batch      = cfg["batch"]
    extra      = cfg.get("extra", {})
    save_dir   = runs_dir / run_name

    print(f"\n{'='*70}")
    print(f"  Training run : {run_name}")
    print(f"  Backbone     : {model_id}")
    print(f"  Epochs       : {epochs}   |  imgsz={imgsz}   |  batch={batch}")
    print(f"  Device       : {device}")
    print(f"{'='*70}\n")

    # ── 1. Configure Ultralytics → MLflow built-in callback ─────────────────
    ul_settings.update({
        "mlflow":      True,
        "tensorboard": False,
        "wandb":       False,
        "clearml":     False,
        "comet":       False,
    })

    # Environment variables consumed by the Ultralytics MLflow callback
    os.environ["MLFLOW_TRACKING_URI"]   = mlflow_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    os.environ["MLFLOW_RUN"]             = f"{run_name}_epochs{epochs}_imgsz{imgsz}"
    # Keep the internal run alive until we query its ID below
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "False"

    # ── 2. Train ─────────────────────────────────────────────────────────────
    model   = YOLO(model_id)
    t_start = time.time()

    try:
        results = model.train(
            data      = str(dataset_yaml),
            epochs    = epochs,
            imgsz     = imgsz,
            batch     = batch,
            device    = device,
            project   = str(runs_dir),
            name      = run_name,
            exist_ok  = True,
            workers   = workers,
            cache     = cache,
            verbose   = True,
            **extra,
        )
    except Exception as exc:
        print(f"\n[ERROR] Training failed for '{run_name}': {exc}")
        return {
            "name": run_name, "model": model_id,
            "status": "FAILED", "error": str(exc),
        }

    elapsed = time.time() - t_start

    # ── 3. Extract metrics ────────────────────────────────────────────────────
    metrics    = extract_results_dict(results)
    per_cls_ap = extract_per_class_ap(save_dir)
    hyp        = parse_hyp(save_dir)

    best_pt  = save_dir / "weights" / "best.pt"
    last_pt  = save_dir / "weights" / "last.pt"
    res_csv  = save_dir / "results.csv"

    # ── 4. Log a summary MLflow run (separate from the Ultralytics run) ───────
    #
    #  We set the tracking URI / experiment here directly (not via env var)
    #  because the Ultralytics callback may have already called mlflow.end_run().
    mlflow.set_tracking_uri(mlflow_uri)
    exp_obj = mlflow.set_experiment(experiment)

    with mlflow.start_run(
        run_name = f"{run_name}__summary",
        tags     = {
            "run_type":  "summary",
            "model":     model_id,
            "run_name":  run_name,
            "epochs":    str(epochs),
            "imgsz":     str(imgsz),
            "batch":     str(batch),
            "device":    device,
            "timestamp": datetime.utcnow().isoformat(),
        },
    ) as summary_run:

        # ---- hyper-parameters -----------------------------------------------
        mlflow.log_params({
            "model":      model_id,
            "epochs":     epochs,
            "imgsz":      imgsz,
            "batch":      batch,
            "device":     device,
            "nc":         NUM_CLASSES,
            "dataset":    str(dataset_yaml),
            "train_time_s": round(elapsed, 1),
        })
        # log any extra YOLO hyps that were read from args.yaml
        if hyp:
            safe_hyp = {
                k: v for k, v in hyp.items()
                if isinstance(v, (int, float, str, bool)) and k not in ("data",)
            }
            mlflow.log_params({f"hyp/{k}": v for k, v in safe_hyp.items()})

        # ---- final scalar metrics -------------------------------------------
        if metrics:
            mlflow.log_metrics(metrics)

        # ---- per-class AP (prefixed to avoid key collision) ─────────────────
        if per_cls_ap:
            mlflow.log_metrics({f"cls_ap/{k}": v for k, v in per_cls_ap.items()})

        # ---- artifacts -------------------------------------------------------
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt),  artifact_path="weights")
        if last_pt.exists():
            mlflow.log_artifact(str(last_pt),  artifact_path="weights")
        if res_csv.exists():
            mlflow.log_artifact(str(res_csv),  artifact_path="results")

        # Log all PNG plots Ultralytics generates (confusion matrix, PR curve …)
        for png in save_dir.glob("*.png"):
            mlflow.log_artifact(str(png), artifact_path="plots")

        summary_run_id = summary_run.info.run_id

    print(f"\n  [✓] Run '{run_name}' complete in {elapsed/60:.1f} min")
    if metrics:
        map50    = metrics.get("metrics/mAP50(B)",    float("nan"))
        map5095  = metrics.get("metrics/mAP50-95(B)", float("nan"))
        prec     = metrics.get("metrics/precision(B)", float("nan"))
        rec      = metrics.get("metrics/recall(B)",    float("nan"))
        fitness  = metrics.get("fitness",               float("nan"))
        print(f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
              f"P={prec:.4f}  R={rec:.4f}  fitness={fitness:.4f}")

    return {
        "name":        run_name,
        "model":       model_id,
        "epochs":      epochs,
        "imgsz":       imgsz,
        "batch":       batch,
        "status":      "OK",
        "train_time_s": round(elapsed, 1),
        "mlflow_run_id": summary_run_id,
        **metrics,
    }


# ===========================================================================
# SECTION 5 – Leaderboard builder
# ===========================================================================

def build_leaderboard(
    results:     list[dict[str, Any]],
    mlflow_uri:  str,
    experiment:  str,
) -> pd.DataFrame:
    """
    Build a sorted leaderboard DataFrame from the list of run summary dicts.
    Logs the table as a CSV artifact on a dedicated 'leaderboard' MLflow run.
    """
    cols = [
        "name", "model", "epochs", "imgsz", "batch",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision(B)", "metrics/recall(B)",
        "fitness", "train_time_s", "status",
    ]

    df = pd.DataFrame(results)
    # Keep only cols that exist in the data
    present = [c for c in cols if c in df.columns]
    df = df[present].copy()

    # Sort by mAP50 (preferred) → fitness → name (fallback when all runs failed)
    _sort_candidates = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]
    sort_col = next((c for c in _sort_candidates if c in df.columns), None)
    if sort_col:
        # Coerce to numeric so string 'nan' / missing don't raise KeyError
        df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
        df.sort_values(sort_col, ascending=False, inplace=True, na_position="last")
    # else: all runs failed and no metric columns exist – keep insertion order
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", df.index + 1)

    # ── Print to stdout ───────────────────────────────────────────────────────
    try:
        from tabulate import tabulate
        print("\n" + "="*80)
        print("  LEADERBOARD")
        print("="*80)
        print(tabulate(df, headers="keys", tablefmt="rounded_outline",
                       floatfmt=".4f", showindex=False))
    except ImportError:
        print("\nLEADERBOARD (install 'tabulate' for nicer formatting):")
        print(df.to_string(index=False))
    print()

    # ── Log as a dedicated MLflow run ─────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="leaderboard", tags={"run_type": "leaderboard"}):
        # Log best-model summary params
        best = df.iloc[0].to_dict() if not df.empty else {}
        mlflow.log_params({
            "best_model":      best.get("name", "n/a"),
            "best_mAP50":      round(float(best.get("metrics/mAP50(B)", 0)), 4),
            "best_mAP50-95":   round(float(best.get("metrics/mAP50-95(B)", 0)), 4),
            "total_runs":      len(results),
            "successful_runs": sum(1 for r in results if r.get("status") == "OK"),
        })

        # Log each model's key metric
        for _, row in df.iterrows():
            if row.get("status") == "OK":
                name = row["name"]
                mlflow.log_metric(f"lb_mAP50_{name}",
                                  float(row.get("metrics/mAP50(B)", 0)))
                mlflow.log_metric(f"lb_mAP5095_{name}",
                                  float(row.get("metrics/mAP50-95(B)", 0)))

        # Save CSV artifact
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        mlflow.log_artifact(
            _bytes_to_tmp(csv_bytes, "leaderboard.csv"),
            artifact_path="leaderboard",
        )

    return df


def _bytes_to_tmp(data: bytes, filename: str) -> str:
    """Write bytes to a temp file and return the path string."""
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / filename
    tmp.write_bytes(data)
    return str(tmp)


# ===========================================================================
# SECTION 6 – MLflow server helper
# ===========================================================================

def start_mlflow_server(uri: str, port: int = 5000) -> None:
    """
    Attempt to launch the MLflow UI in the background (non-blocking).
    Prints the URL whether or not the launch succeeds.
    """
    import subprocess
    cmd = ["mlflow", "ui", "--backend-store-uri", uri, "--port", str(port)]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n[MLflow UI] Launching: {' '.join(cmd)}")
        print(f"[MLflow UI] Open → http://127.0.0.1:{port}\n")
    except FileNotFoundError:
        print(f"\n[MLflow UI] mlflow CLI not on PATH. Start manually:\n"
              f"    mlflow ui --backend-store-uri {uri} --port {port}\n")


# ===========================================================================
# SECTION 7 – Main orchestrator
# ===========================================================================

def run_sweep(
    dataset_dir:  Path,
    runs_dir:     Path,
    experiment:   str,
    mlflow_uri:   str,
    device:       str,
    workers:      int,
    cache:        bool,
    sweep:        list[dict],
    launch_ui:    bool,
    ui_port:      int,
) -> None:
    """Execute the full multi-model sweep and build the leaderboard."""

    # ── Resolve dataset ───────────────────────────────────────────────────────
    dataset_yaml = resolve_dataset_yaml(dataset_dir)
    print(f"\n[i] Dataset YAML : {dataset_yaml}")
    print(f"[i] MLflow URI   : {mlflow_uri}")
    print(f"[i] Experiment   : {experiment}")
    print(f"[i] Runs dir     : {runs_dir}")
    print(f"[i] Models       : {[c['name'] for c in sweep]}")
    print(f"[i] Total runs   : {len(sweep)}\n")

    runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Optionally start MLflow UI in background ──────────────────────────────
    if launch_ui:
        start_mlflow_server(mlflow_uri, ui_port)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for i, cfg in enumerate(sweep, start=1):
        print(f"\n[{i}/{len(sweep)}] Starting run: {cfg['name']}")
        result = train_one(
            cfg          = cfg,
            dataset_yaml = dataset_yaml,
            runs_dir     = runs_dir,
            device       = device,
            experiment   = experiment,
            mlflow_uri   = mlflow_uri,
            workers      = workers,
            cache        = cache,
        )
        all_results.append(result)

        # Persist intermediate results as JSON so a crash doesn't lose data
        interim_path = runs_dir / "sweep_results.json"
        with open(interim_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    print("\n[i] Building leaderboard …")
    leaderboard = build_leaderboard(all_results, mlflow_uri, experiment)

    # Persist final leaderboard CSV locally
    lb_path = runs_dir / "leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)
    print(f"[✓] Leaderboard saved → {lb_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    best = leaderboard.iloc[0] if not leaderboard.empty else {}
    print("\n" + "="*60)
    print("  SWEEP COMPLETE")
    print("="*60)
    print(f"  Runs: {len(all_results)} total  |  "
          f"{sum(1 for r in all_results if r.get('status')=='OK')} OK  |  "
          f"{sum(1 for r in all_results if r.get('status')!='OK')} failed")
    if best.get("name"):
        print(f"  Best model : {best['name']}")
        print(f"  mAP50      : {best.get('metrics/mAP50(B)', float('nan')):.4f}")
        print(f"  mAP50-95   : {best.get('metrics/mAP50-95(B)', float('nan')):.4f}")
    print(f"\n  MLflow UI  : http://127.0.0.1:{ui_port}")
    print(f"  Tracking   : {mlflow_uri}")
    print("="*60 + "\n")


# ===========================================================================
# SECTION 8 – CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model YOLO training sweep with MLflow tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # use default 6-model sweep on CPU
  python train_yolo_mlflow.py --dataset-dir ./output

  # GPU, custom experiment name, fewer epochs (quick test)
  python train_yolo_mlflow.py \\
      --dataset-dir ./output \\
      --experiment  khmer-id-v2 \\
      --device      0 \\
      --epochs      10 \\
      --models      yolo11n yolo11s

  # point at a remote MLflow server
  python train_yolo_mlflow.py \\
      --dataset-dir ./output \\
      --mlflow-uri  http://my-mlflow-server:5000

  # then browse results
  mlflow ui --backend-store-uri ./mlruns
""",
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        default="./output",
        help="Root of the generate_synthetic_id.py output directory (default: ./output).",
    )
    parser.add_argument(
        "--runs-dir", "-r",
        default="./runs",
        help="Directory where Ultralytics saves weights / plots (default: ./runs).",
    )
    parser.add_argument(
        "--experiment", "-e",
        default="khmer-id-detection",
        help="MLflow experiment name (default: khmer-id-detection).",
    )
    parser.add_argument(
        "--mlflow-uri", "-u",
        default="./mlruns",
        help="MLflow tracking URI – local path or http://host:port (default: ./mlruns).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device: 'cpu', '0', '0,1', 'mps' (default: cpu).",
    )
    parser.add_argument(
        "--workers",
        type=int, default=4,
        help="DataLoader worker processes (default: 4). Use 0 on Windows.",
    )
    parser.add_argument(
        "--cache",
        action="store_true", default=False,
        help="Cache images in RAM for faster training (needs enough memory).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Subset of run names from the sweep to execute. "
            "E.g. --models yolo11n yolo11s  "
            "Available: " + ", ".join(c["name"] for c in default_sweep())
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int, default=None,
        help="Override epochs for ALL models in the sweep.",
    )
    parser.add_argument(
        "--batch",
        type=int, default=None,
        help="Override batch size for ALL models in the sweep.",
    )
    parser.add_argument(
        "--imgsz",
        type=int, default=None,
        help="Override image size for ALL models in the sweep.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true", default=False,
        help="Do not launch the MLflow UI in the background.",
    )
    parser.add_argument(
        "--ui-port",
        type=int, default=5000,
        help="Port for the background MLflow UI server (default: 5000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build sweep ───────────────────────────────────────────────────────────
    sweep = default_sweep()

    # Filter by requested model names
    if args.models:
        sweep = [c for c in sweep if c["name"] in args.models]
        if not sweep:
            sys.exit(
                f"[ERROR] None of the requested models matched.\n"
                f"Available: {[c['name'] for c in default_sweep()]}"
            )

    # Apply global CLI overrides
    for cfg in sweep:
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        if args.batch is not None:
            cfg["batch"] = args.batch
        if args.imgsz is not None:
            cfg["imgsz"] = args.imgsz

    # ── Resolve paths ─────────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir).resolve()
    runs_dir    = Path(args.runs_dir).resolve()
    mlflow_uri  = args.mlflow_uri

    # Convert any local path to an explicit file:/// URI.
    # This fixes Windows backslash / space issues and satisfies MLflow's
    # URI scheme validation (plain absolute paths are rejected on Windows).
    if not mlflow_uri.startswith("http"):
        mlflow_uri = Path(mlflow_uri).resolve().as_uri()  # file:///D:/project/mlruns

    # ── Run ───────────────────────────────────────────────────────────────────
    run_sweep(
        dataset_dir  = dataset_dir,
        runs_dir     = runs_dir,
        experiment   = args.experiment,
        mlflow_uri   = mlflow_uri,
        device       = args.device,
        workers      = args.workers,
        cache        = args.cache,
        sweep        = sweep,
        launch_ui    = not args.no_ui,
        ui_port      = args.ui_port,
    )


if __name__ == "__main__":
    main()