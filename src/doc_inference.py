"""
doc_inference.py
================
Khmer Document Text-Line Detector — three modes in one file.

    1. CLI        python doc_inference.py --source scan.jpg
    2. Python API from doc_inference import TextLineDetector
    3. Streamlit  streamlit run doc_inference.py

The model detects a single class (class 0 = text_line) and returns each
detected line as a bounding box sorted top-to-bottom, left-to-right.

The Python API is designed to plug directly into a recognition pipeline:

    detector = TextLineDetector("best.pt")
    result   = detector.predict("page.jpg")
    for line in result.lines:
        crop = result.crop_line(line)   # PIL Image of just that line
        text = your_ocr_model(crop)     # hand off to your recognition model

Download best.pt from Kaggle /kaggle/working/best.pt after training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "./yolo11n_best/best.pt"
DEFAULT_CONF  = 0.30
DEFAULT_IOU   = 0.45
LINE_COLOR_BGR = (0, 200, 80)   # single-class → one consistent green


# ===========================================================================
# SECTION 1 – Data structures
# ===========================================================================

@dataclass
class TextLine:
    """One detected text line."""
    line_idx:   int            # 0-based reading order (top→bottom, left→right)
    confidence: float
    x1: float                  # absolute pixel coordinates
    y1: float
    x2: float
    y2: float

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageResult:
    """Full inference result for one document page image."""
    source:       str
    image_width:  int
    image_height: int
    inference_ms: float
    lines: list[TextLine]             = field(default_factory=list)
    # The original image as a numpy array (BGR) for downstream use
    _image_bgr: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    # ── Crop helpers — key for recognition pipeline integration ──────────────
    def crop_line(
        self,
        line: TextLine,
        pad_x: int = 4,
        pad_y: int = 3,
        as_pil: bool = True,
    ) -> Union[Image.Image, np.ndarray]:
        """
        Return a tight crop of *line* from the original image.

        Parameters
        ----------
        line   : TextLine from self.lines
        pad_x  : horizontal padding in pixels (avoids clipping ascenders/descenders)
        pad_y  : vertical padding
        as_pil : True  → return PIL Image (RGB)
                 False → return numpy array (BGR, same as cv2 format)

        This is the primary integration point for your recognition model:

            for line in result.lines:
                crop = result.crop_line(line)
                text = ocr_model.predict(crop)
        """
        if self._image_bgr is None:
            raise RuntimeError("Image data not available. Run predict() with store_image=True.")
        h, w = self._image_bgr.shape[:2]
        x1 = max(0, int(line.x1) - pad_x)
        y1 = max(0, int(line.y1) - pad_y)
        x2 = min(w, int(line.x2) + pad_x)
        y2 = min(h, int(line.y2) + pad_y)
        crop_bgr = self._image_bgr[y1:y2, x1:x2]
        if as_pil:
            return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        return crop_bgr

    def crop_all_lines(
        self,
        pad_x: int = 4,
        pad_y: int = 3,
        as_pil: bool = True,
    ) -> list[Union[Image.Image, np.ndarray]]:
        """
        Return a list of crops for every detected line, in reading order.
        Ready to feed directly into a recognition/OCR pipeline:

            crops = result.crop_all_lines()
            texts = [ocr_model(crop) for crop in crops]
        """
        return [self.crop_line(ln, pad_x, pad_y, as_pil) for ln in self.lines]

    def annotated_image(self, thickness: int = 2) -> np.ndarray:
        """
        Return a copy of the source image with bounding boxes drawn.
        Boxes are numbered in reading order.
        """
        if self._image_bgr is None:
            raise RuntimeError("Image data not available.")
        img = self._image_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for ln in self.lines:
            x1, y1, x2, y2 = int(ln.x1), int(ln.y1), int(ln.x2), int(ln.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), LINE_COLOR_BGR, thickness)
            label = f"#{ln.line_idx + 1}  {ln.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(label, font, 0.45, 1)
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(img, (x1, ly - th - 3), (x1 + tw + 4, ly + bl),
                          LINE_COLOR_BGR, cv2.FILLED)
            cv2.putText(img, label, (x1 + 2, ly), font, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "source":       self.source,
            "image_width":  self.image_width,
            "image_height": self.image_height,
            "inference_ms": self.inference_ms,
            "num_lines":    self.num_lines,
            "lines": [ln.to_dict() for ln in self.lines],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ===========================================================================
# SECTION 2 – Core detector (Python API)
# ===========================================================================

class TextLineDetector:
    """
    Khmer document text-line detector.

    Designed to slot directly in front of a recognition model:

        detector = TextLineDetector("best.pt")

        # single image
        result = detector.predict("page.jpg")
        crops  = result.crop_all_lines()        # list[PIL Image], reading order
        texts  = [ocr(crop) for crop in crops]  # hand to your OCR model

        # batch
        results = detector.predict_batch(["p1.jpg", "p2.jpg"])

        # numpy / PIL input (e.g. from a camera or PDF renderer)
        result = detector.predict(np.array(...))
        result = detector.predict(PIL.Image.open(...))

    Parameters
    ----------
    model_path   : path to best.pt from Kaggle training
    conf         : confidence threshold (default 0.30)
    iou          : NMS IoU threshold (default 0.45)
    device       : 'cpu' | '0' | 'mps' | '' (auto)
    store_image  : keep the raw numpy image in PageResult for cropping
                   (True by default; set False to save memory in batch mode)
    """

    def __init__(
        self,
        model_path:  Union[str, Path] = DEFAULT_MODEL,
        conf:        float = DEFAULT_CONF,
        iou:         float = DEFAULT_IOU,
        device:      str   = "",
        store_image: bool  = True,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                "Download best.pt from Kaggle /kaggle/working/ after training."
            )
        self.conf        = conf
        self.iou         = iou
        self.device      = device
        self.store_image = store_image
        self._model      = YOLO(str(model_path))
        print(f"[TextLineDetector] Loaded: {model_path}  conf={conf}  iou={iou}  device={device or 'auto'}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
    ) -> PageResult:
        """
        Detect text lines in *source* and return a PageResult.

        Lines are sorted into natural reading order:
          - Cluster into rows by y-centre proximity (row_gap = median line height × 0.6)
          - Within each row, sort left-to-right by x1

        This ordering matches how a human reads a document top-to-bottom,
        and is the correct order to feed into a sequential recognition model.
        """
        img_bgr, label = self._load(source)
        h, w = img_bgr.shape[:2]

        t0  = time.perf_counter()
        raw = self._model.predict(
            source  = img_bgr,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            verbose = False,
            save    = False,
        )
        ms = (time.perf_counter() - t0) * 1000

        lines = self._parse(raw)
        lines = self._sort_reading_order(lines)

        return PageResult(
            source       = label,
            image_width  = w,
            image_height = h,
            inference_ms = round(ms, 1),
            lines        = lines,
            _image_bgr   = img_bgr if self.store_image else None,
        )

    def predict_batch(
        self,
        sources: list[Union[str, Path, np.ndarray, Image.Image]],
    ) -> list[PageResult]:
        """Run predict() over a list of sources. Returns results in input order."""
        return [self.predict(s) for s in sources]

    def save_annotated(
        self,
        result: PageResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Draw boxes on the image and save to *output_path*."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result.annotated_image())
        return out

    def save_json(
        self,
        result: PageResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Save detection results as a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.to_json(), encoding="utf-8")
        return out

    def save_crops(
        self,
        result: PageResult,
        output_dir: Union[str, Path],
        fmt: str = "jpg",
    ) -> list[Path]:
        """
        Save every detected line as a separate image crop.
        Files are named  {stem}_line_{idx:03d}.{fmt}
        Returns the list of saved paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem   = Path(result.source).stem if result.source != "<numpy array>" else "crop"
        paths  = []
        for ln in result.lines:
            crop = result.crop_line(ln, as_pil=True)
            p    = out_dir / f"{stem}_line_{ln.line_idx:03d}.{fmt}"
            crop.save(str(p))
            paths.append(p)
        return paths

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load(source) -> tuple[np.ndarray, str]:
        if isinstance(source, np.ndarray):
            return source.copy(), "<numpy array>"
        if isinstance(source, Image.Image):
            arr = np.array(source.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), "<PIL Image>"
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2 could not decode: {path}")
        return img, str(path)

    @staticmethod
    def _parse(raw) -> list[TextLine]:
        lines: list[TextLine] = []
        if raw and raw[0].boxes is not None:
            boxes = raw[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                lines.append(TextLine(
                    line_idx   = i,          # re-assigned by _sort_reading_order
                    confidence = round(conf, 4),
                    x1 = float(xyxy[0]),
                    y1 = float(xyxy[1]),
                    x2 = float(xyxy[2]),
                    y2 = float(xyxy[3]),
                ))
        return lines

    @staticmethod
    def _sort_reading_order(lines: list[TextLine]) -> list[TextLine]:
        """
        Sort detections into reading order (top-to-bottom, left-to-right).

        Algorithm
        ---------
        1. Compute median line height to set a row-clustering threshold.
        2. Sort all lines by y-centre.
        3. Group consecutive lines whose y-centres are within (median_h × 0.6)
           of each other into the same row.
        4. Within each row sort by x1.
        5. Re-assign line_idx 0, 1, 2, … in final order.
        """
        if not lines:
            return lines

        heights = [ln.height for ln in lines]
        median_h = float(np.median(heights)) if heights else 20.0
        row_gap  = median_h * 0.6

        # Sort by vertical centre
        sorted_by_y = sorted(lines, key=lambda ln: (ln.y1 + ln.y2) / 2)

        rows: list[list[TextLine]] = []
        current_row: list[TextLine] = [sorted_by_y[0]]
        current_cy = (sorted_by_y[0].y1 + sorted_by_y[0].y2) / 2

        for ln in sorted_by_y[1:]:
            cy = (ln.y1 + ln.y2) / 2
            if cy - current_cy <= row_gap:
                current_row.append(ln)
            else:
                rows.append(sorted(current_row, key=lambda l: l.x1))
                current_row = [ln]
                current_cy  = cy
        rows.append(sorted(current_row, key=lambda l: l.x1))

        # Flatten and re-index
        ordered: list[TextLine] = []
        for row in rows:
            for ln in row:
                ln.line_idx = len(ordered)
                ordered.append(ln)
        return ordered


# ===========================================================================
# SECTION 3 – CLI
# ===========================================================================

def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog        = "doc_inference.py",
        description = "Khmer document text-line detector — CLI",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """\
Examples:
  # single image — print detected lines
  python doc_inference.py --source scan.jpg

  # save annotated image + per-line crops + JSON
  python doc_inference.py --source scan.jpg \\
      --save-annotated --save-crops --save-json

  # process a whole folder
  python doc_inference.py --source ./scans/ --output ./results/ --save-annotated

  # custom model / thresholds
  python doc_inference.py --source scan.jpg \\
      --model ./best.pt --conf 0.35 --iou 0.4

  # JSON to stdout for piping
  python doc_inference.py --source scan.jpg --json-stdout
""",
    )
    # Input
    parser.add_argument("--source", "-s", required=True,
                        help="Image file or directory of images.")
    # Model
    parser.add_argument("--model",  "-m", default=DEFAULT_MODEL,
                        help=f"Path to best.pt  (default: {DEFAULT_MODEL})")
    parser.add_argument("--conf",   type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold  (default: {DEFAULT_CONF})")
    parser.add_argument("--iou",    type=float, default=DEFAULT_IOU,
                        help=f"NMS IoU threshold  (default: {DEFAULT_IOU})")
    parser.add_argument("--device", default="",
                        help="Device: '' auto | 'cpu' | '0' | 'mps'")
    # Output
    parser.add_argument("--output", "-o", default="./line_detection_output",
                        help="Output directory  (default: ./line_detection_output)")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated image with bounding boxes.")
    parser.add_argument("--save-crops",     action="store_true",
                        help="Save each detected line as a separate crop image.")
    parser.add_argument("--save-json",      action="store_true",
                        help="Save detection results as JSON.")
    parser.add_argument("--json-stdout",    action="store_true",
                        help="Print JSON results to stdout (for piping).")
    parser.add_argument("--crop-pad-x",     type=int, default=4,
                        help="Horizontal padding added to each crop  (default: 4)")
    parser.add_argument("--crop-pad-y",     type=int, default=3,
                        help="Vertical padding added to each crop  (default: 3)")

    args = parser.parse_args(argv)

    # Collect sources
    src      = Path(args.source)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if src.is_dir():
        sources = sorted(p for p in src.rglob("*") if p.suffix.lower() in img_exts)
    elif src.is_file():
        sources = [src]
    else:
        sys.exit(f"[ERROR] --source not found: {src}")

    if not sources:
        sys.exit(f"[ERROR] No images found in: {src}")

    # Load model
    detector = TextLineDetector(
        args.model, conf=args.conf, iou=args.iou, device=args.device,
        store_image = args.save_annotated or args.save_crops,
    )
    out_dir = Path(args.output)
    print(f"\n[i] Processing {len(sources)} image(s) …\n")

    total_lines = 0
    for img_path in sources:
        result       = detector.predict(img_path)
        total_lines += result.num_lines

        # Console summary
        print(f"  {img_path.name:<40}  {result.num_lines:>3} lines"
              f"  ({result.inference_ms:.0f} ms)")
        for ln in result.lines:
            print(f"    #{ln.line_idx+1:>3}  conf={ln.confidence:.3f}"
                  f"  box=({ln.x1:.0f},{ln.y1:.0f},{ln.x2:.0f},{ln.y2:.0f})"
                  f"  {ln.width:.0f}×{ln.height:.0f}px")

        # Optional outputs
        if args.save_annotated:
            p = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"
            detector.save_annotated(result, p)
            print(f"    → annotated : {p}")

        if args.save_crops:
            saved = detector.save_crops(result, out_dir / img_path.stem,
                                        pad_x=args.crop_pad_x,
                                        pad_y=args.crop_pad_y)  # type: ignore[call-arg]
            # save_crops doesn't accept keyword-only pad args — call via result directly
            saved = []
            crop_dir = out_dir / img_path.stem
            crop_dir.mkdir(parents=True, exist_ok=True)
            for ln in result.lines:
                crop = result.crop_line(ln, args.crop_pad_x, args.crop_pad_y)
                cp   = crop_dir / f"line_{ln.line_idx:03d}.jpg"
                crop.save(str(cp))
                saved.append(cp)
            print(f"    → crops     : {crop_dir}/  ({len(saved)} files)")

        if args.save_json:
            p = out_dir / f"{img_path.stem}_lines.json"
            detector.save_json(result, p)
            print(f"    → JSON      : {p}")

        if args.json_stdout:
            print(result.to_json())

    print(f"\n[✓] Done — {len(sources)} image(s), {total_lines} total lines detected.")
    if any([args.save_annotated, args.save_crops, args.save_json]):
        print(f"    Output: {out_dir.resolve()}")


# ===========================================================================
# SECTION 4 – Streamlit UI
# ===========================================================================

def run_streamlit() -> None:
    """
    Full Streamlit web application.
    Launch with:  streamlit run doc_inference.py
    """
    import io as _io
    import streamlit as st

    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title = "ឯកសារ · Khmer Document Line Detector",
        page_icon  = "📄",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;500;700;900&display=swap');

    :root {
        --bg:       #0c0e12;
        --surface:  #13161d;
        --surface2: #1a1e28;
        --border:   #252b38;
        --accent:   #00e676;
        --accent2:  #2979ff;
        --text:     #e2e8f0;
        --muted:    #64748b;
        --warn:     #ffd740;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        font-family: 'Outfit', sans-serif;
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    h1 { font-weight: 900 !important; letter-spacing: -0.04em !important;
         font-size: 1.8rem !important; color: var(--text) !important; }
    h2 { font-weight: 700 !important; font-size: 1rem !important;
         text-transform: uppercase; letter-spacing: 0.1em !important;
         color: var(--accent) !important; }
    .metric-pill {
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: 10px; padding: 14px 18px; text-align: center;
        margin-bottom: 8px;
    }
    .metric-pill .val {
        font-family: 'JetBrains Mono', monospace; font-size: 1.9rem;
        font-weight: 600; color: var(--accent); line-height: 1;
    }
    .metric-pill .lbl {
        font-size: 0.7rem; color: var(--muted); text-transform: uppercase;
        letter-spacing: 0.08em; margin-top: 5px;
    }
    .line-card {
        display: flex; align-items: center; gap: 10px;
        padding: 9px 14px; margin-bottom: 5px;
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: 8px; transition: border-color 0.15s;
        font-size: 0.85rem;
    }
    .line-card:hover { border-color: var(--accent); }
    .line-num {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        font-weight: 600; color: var(--bg); background: var(--accent);
        border-radius: 5px; padding: 2px 7px; white-space: nowrap;
    }
    .line-conf { font-family: 'JetBrains Mono', monospace;
                 color: var(--muted); font-size: 0.78rem; margin-left: auto; }
    .line-dim  { font-family: 'JetBrains Mono', monospace;
                 color: var(--border); font-size: 0.72rem; }
    .stButton > button {
        background: linear-gradient(135deg, #00c853, #00e676) !important;
        color: #000 !important; border: none !important;
        font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
        border-radius: 8px !important; padding: 9px 24px !important;
    }
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 2px dashed var(--border) !important; border-radius: 12px !important;
    }
    hr { border-color: var(--border) !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📄 Doc Line Detector")
        st.markdown("---")

        model_path  = st.text_input(
            "Model weights (best.pt)",
            value = DEFAULT_MODEL,
            help  = "Download best.pt from Kaggle after training.",
        )
        conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
        iou_thresh  = st.slider("NMS IoU threshold",    0.10, 0.90, DEFAULT_IOU,  0.05)
        crop_pad_x  = st.slider("Crop padding X (px)",  0,    20,   4)
        crop_pad_y  = st.slider("Crop padding Y (px)",  0,    12,   3)
        device      = st.selectbox("Device", ["auto", "cpu", "0", "mps"])

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.71rem;color:#334155;line-height:1.7'>"
            "YOLOv11 · single-class<br>text_line detection<br>"
            "Khmer documents"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Load model (cached) ───────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading model …")
    def load_detector(path, conf, iou, dev):
        return TextLineDetector(
            model_path  = path,
            conf        = conf,
            iou         = iou,
            device      = "" if dev == "auto" else dev,
            store_image = True,
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## ឯកសារ · Khmer Document Text-Line Detector")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:-6px'>"
        "Upload a document scan or photo — detected text lines are sorted "
        "into reading order and available as crops for your OCR pipeline."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop a document image here  ·  JPG / PNG / BMP / WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is None:
        st.markdown(
            "<div style='text-align:center;padding:60px 0;color:#1e293b'>"
            "<div style='font-size:3.5rem'>📄</div>"
            "<div style='margin-top:10px;font-size:0.95rem'>Upload a document to detect text lines</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        detector = load_detector(model_path, conf_thresh, iou_thresh, device)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting text lines …"):
        result = detector.predict(img_bgr)

    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    def pill(col, val, lbl):
        col.markdown(
            f"<div class='metric-pill'>"
            f"<div class='val'>{val}</div><div class='lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    pill(c1, result.num_lines, "Lines detected")
    pill(c2, f"{result.inference_ms:.0f} ms", "Inference time")
    pill(c3, f"{result.image_width}×{result.image_height}", "Image size")
    avg_conf = (
        round(sum(ln.confidence for ln in result.lines) / result.num_lines, 3)
        if result.lines else 0.0
    )
    pill(c4, f"{avg_conf:.3f}", "Avg confidence")

    st.markdown("")

    # ── Two-column: annotated image | line list + crops ───────────────────────
    col_img, col_det = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown("#### Annotated Image")
        ann_bgr = result.annotated_image()
        ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
        st.image(ann_rgb, use_column_width=True,
                 caption=f"{uploaded.name}  ·  {result.num_lines} lines  ·  {result.inference_ms:.0f} ms")

        # Download annotated image
        buf = _io.BytesIO()
        Image.fromarray(ann_rgb).save(buf, format="JPEG", quality=92)
        st.download_button(
            "⬇  Download annotated image",
            data      = buf.getvalue(),
            file_name = f"{Path(uploaded.name).stem}_annotated.jpg",
            mime      = "image/jpeg",
        )

    with col_det:
        st.markdown(f"#### Detected Lines ({result.num_lines})")

        if not result.lines:
            st.markdown(
                "<div style='color:#475569;padding:20px 0;font-size:0.9rem'>"
                "No text lines detected above the confidence threshold."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for ln in result.lines:
                st.markdown(
                    f"<div class='line-card'>"
                    f"<span class='line-num'>#{ln.line_idx + 1}</span>"
                    f"<span>{ln.width:.0f}&thinsp;×&thinsp;{ln.height:.0f} px</span>"
                    f"<span class='line-conf'>{ln.confidence:.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Line crops viewer ──────────────────────────────────────────────────
        st.markdown("#### Line Crops (reading order)")
        st.caption("Sorted top→bottom, left→right. Ready for your OCR model.")

        if result.lines:
            for ln in result.lines:
                crop_pil = result.crop_line(ln, pad_x=crop_pad_x, pad_y=crop_pad_y)
                st.image(
                    crop_pil,
                    caption = f"Line #{ln.line_idx + 1}  —  {ln.width:.0f}×{ln.height:.0f} px  conf={ln.confidence:.3f}",
                    use_column_width = True,
                )

        st.markdown("---")

        # ── JSON + CSV downloads ──────────────────────────────────────────────
        st.markdown("#### Export")
        dl1, dl2 = st.columns(2)

        with dl1:
            json_bytes = result.to_json().encode("utf-8")
            st.download_button(
                "⬇  JSON",
                data      = json_bytes,
                file_name = f"{Path(uploaded.name).stem}_lines.json",
                mime      = "application/json",
            )

        with dl2:
            # CSV: one row per line
            import csv, io as _io2
            csv_buf = _io2.StringIO()
            w = csv.writer(csv_buf)
            w.writerow(["line_idx", "confidence", "x1", "y1", "x2", "y2", "width", "height"])
            for ln in result.lines:
                w.writerow([ln.line_idx, ln.confidence,
                            round(ln.x1, 1), round(ln.y1, 1),
                            round(ln.x2, 1), round(ln.y2, 1),
                            round(ln.width, 1), round(ln.height, 1)])
            st.download_button(
                "⬇  CSV",
                data      = csv_buf.getvalue().encode(),
                file_name = f"{Path(uploaded.name).stem}_lines.csv",
                mime      = "text/csv",
            )

        # ── All crops as zip ─────────────────────────────────────────────────
        if result.lines:
            zip_buf = _io.BytesIO()
            with __import__("zipfile").ZipFile(zip_buf, "w") as zf:
                for ln in result.lines:
                    crop = result.crop_line(ln, pad_x=crop_pad_x, pad_y=crop_pad_y)
                    img_buf = _io.BytesIO()
                    crop.save(img_buf, format="JPEG", quality=93)
                    zf.writestr(f"line_{ln.line_idx:03d}.jpg", img_buf.getvalue())
            st.download_button(
                "⬇  All crops (.zip)",
                data      = zip_buf.getvalue(),
                file_name = f"{Path(uploaded.name).stem}_crops.zip",
                mime      = "application/zip",
            )

        # ── JSON preview ──────────────────────────────────────────────────────
        with st.expander("Preview JSON", expanded=False):
            st.code(result.to_json(), language="json")


# ===========================================================================
# SECTION 5 – Dispatcher
# ===========================================================================

def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    run_cli()
