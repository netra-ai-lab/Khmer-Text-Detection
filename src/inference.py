"""
inference.py
============
Khmer National ID Card – YOLO Text Field Detector
Three inference modes in a single file:

    1. CLI          python inference.py --source image.jpg
    2. Python API   from inference import IDCardDetector; det = IDCardDetector(...)
    3. Streamlit    streamlit run inference.py

Class map (12 classes):
    0  id_number       7  validity
    1  name_kh         8  features
    2  name_en         9  mrz_1
    3  dob_sex_height  10  mrz_2
    4  pob             11  mrz_3
    5  address_1
    6  address_2
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
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "id_number", "name_kh", "name_en", "dob_sex_height",
    "pob", "address_1", "address_2", "validity",
    "features", "mrz_1", "mrz_2", "mrz_3",
]

# Distinct BGR colours for each class (used in cv2 drawing)
CLASS_COLORS_BGR: list[tuple[int, int, int]] = [
    (0,   200, 255),   # 0  id_number       – amber
    (50,  205,  50),   # 1  name_kh          – lime green
    (255, 140,   0),   # 2  name_en          – deep sky blue
    (147,  20, 255),   # 3  dob_sex_height   – purple
    (0,   165, 255),   # 4  pob              – orange
    (220,  20,  60),   # 5  address_1        – crimson
    (255,   0, 144),   # 6  address_2        – hot pink
    (0,   255, 195),   # 7  validity         – cyan-green
    (30,  144, 255),   # 8  features         – dodger blue
    (255,  69,   0),   # 9  mrz_1            – orange-red
    (186,  85, 211),   # 10 mrz_2            – medium orchid
    (60,  179, 113),   # 11 mrz_3            – medium sea green
]

DEFAULT_MODEL  = "./runs/yolo11n/weights/best.pt"
DEFAULT_CONF   = 0.25
DEFAULT_IOU    = 0.45
CARD_W, CARD_H = 1000, 630


# ===========================================================================
# SECTION 1 – Data structures
# ===========================================================================

@dataclass
class Detection:
    """Single detected text field."""
    class_id:   int
    class_name: str
    confidence: float
    # Bounding box in absolute pixels  (x_min, y_min, x_max, y_max)
    x1: float
    y1: float
    x2: float
    y2: float

    # Convenience properties
    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, width, height) in pixels."""
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceResult:
    """Full result for one image."""
    source:          str
    image_width:     int
    image_height:    int
    inference_ms:    float
    detections:      list[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        """Serialisable dict (excludes the numpy image array)."""
        return {
            "source":       self.source,
            "image_width":  self.image_width,
            "image_height": self.image_height,
            "inference_ms": self.inference_ms,
            "num_detections": self.num_detections,
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ===========================================================================
# SECTION 2 – Core detector (Python API)
# ===========================================================================

class IDCardDetector:
    """
    Khmer National ID card text-field detector.

    Parameters
    ----------
    model_path : str | Path
        Path to a trained best.pt weights file.
    conf       : float
        Confidence threshold (0–1).  Default 0.25.
    iou        : float
        IoU threshold for NMS.  Default 0.45.
    device     : str
        'cpu', '0', 'mps', etc.  Empty string = auto.

    Examples
    --------
    >>> det = IDCardDetector("./runs/yolo11n/weights/best.pt")
    >>> result = det.predict("card.jpg")
    >>> print(result.to_json())
    >>> det.save_annotated(result, "out.jpg")
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL,
        conf:   float = DEFAULT_CONF,
        iou:    float = DEFAULT_IOU,
        device: str   = "",
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                "Run train_yolo_mlflow.py first, or pass the correct path."
            )
        self.conf   = conf
        self.iou    = iou
        self.device = device
        self._model = YOLO(str(model_path))
        print(f"[IDCardDetector] Loaded: {model_path}  conf={conf}  iou={iou}")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        annotate: bool = True,
    ) -> InferenceResult:
        """
        Run detection on *source* and return an InferenceResult.

        Parameters
        ----------
        source   : file path | numpy (H,W,3) BGR | PIL Image
        annotate : if True, draw coloured boxes onto the image and store in
                   result.annotated_image (numpy BGR array).
        """
        # ── Load image ────────────────────────────────────────────────────────
        img_bgr, source_label = self._load_image(source)
        h, w = img_bgr.shape[:2]

        # ── Inference ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        raw = self._model.predict(
            source      = img_bgr,
            conf        = self.conf,
            iou         = self.iou,
            device      = self.device,
            verbose     = False,
            save        = False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Parse detections ──────────────────────────────────────────────────
        detections: list[Detection] = []
        if raw and raw[0].boxes is not None:
            boxes = raw[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf_val  = float(boxes.conf[i].cpu().numpy())
                cls_id    = int(boxes.cls[i].cpu().numpy())
                cls_name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                detections.append(Detection(
                    class_id   = cls_id,
                    class_name = cls_name,
                    confidence = round(conf_val, 4),
                    x1 = float(xyxy[0]),
                    y1 = float(xyxy[1]),
                    x2 = float(xyxy[2]),
                    y2 = float(xyxy[3]),
                ))

        # Sort by y1 so detections read top-to-bottom (natural reading order)
        detections.sort(key=lambda d: d.y1)

        # ── Build annotated image ─────────────────────────────────────────────
        annotated = self._draw_boxes(img_bgr.copy(), detections) if annotate else None

        return InferenceResult(
            source          = source_label,
            image_width     = w,
            image_height    = h,
            inference_ms    = round(elapsed_ms, 1),
            detections      = detections,
            annotated_image = annotated,
        )

    def predict_batch(
        self,
        sources: list[Union[str, Path, np.ndarray, Image.Image]],
        annotate: bool = True,
    ) -> list[InferenceResult]:
        """Run predict() over a list of sources and return a list of results."""
        return [self.predict(s, annotate=annotate) for s in sources]

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------

    def save_annotated(self, result: InferenceResult, output_path: Union[str, Path]) -> Path:
        """Write result.annotated_image to *output_path* (JPEG/PNG)."""
        if result.annotated_image is None:
            raise ValueError("No annotated image in result. Re-run predict(annotate=True).")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result.annotated_image)
        return out

    def save_json(self, result: InferenceResult, output_path: Union[str, Path]) -> Path:
        """Write detection results as JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.to_json(), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(
        source: Union[str, Path, np.ndarray, Image.Image],
    ) -> tuple[np.ndarray, str]:
        """Return (BGR numpy array, label string)."""
        if isinstance(source, np.ndarray):
            # Accept both BGR (cv2) and RGB arrays
            return source.copy(), "<numpy array>"

        if isinstance(source, Image.Image):
            arr = np.array(source.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), "<PIL Image>"

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2 could not read image: {path}")
        return img, str(path)

    @staticmethod
    def _draw_boxes(
        img: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw annotated bounding boxes + labels onto *img* (in-place)."""
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness  = 2

        for det in detections:
            color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Label background + text
            label      = f"{det.class_name}  {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            label_y    = max(y1 - 6, th + 4)
            cv2.rectangle(img,
                          (x1, label_y - th - 4),
                          (x1 + tw + 4, label_y + baseline),
                          color, cv2.FILLED)
            cv2.putText(img, label,
                        (x1 + 2, label_y),
                        font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return img


# ===========================================================================
# SECTION 3 – CLI
# ===========================================================================

def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog        = "inference.py",
        description = "Khmer ID card text-field detector – CLI mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # single image, save annotated output + JSON
  python inference.py --source card.jpg --save-image --save-json

  # directory of images
  python inference.py --source ./test_images/ --output ./results/

  # custom model / thresholds
  python inference.py --source card.jpg --model ./runs/yolo11n/weights/best.pt --conf 0.4

  # print JSON to stdout (pipe-friendly)
  python inference.py --source card.jpg --json-stdout
""",
    )

    # Input
    parser.add_argument("--source", "-s", required=True,
                        help="Image file or directory of images.")
    # Model
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Path to best.pt weights (default: {DEFAULT_MODEL}).")
    parser.add_argument("--conf",  type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF}).")
    parser.add_argument("--iou",   type=float, default=DEFAULT_IOU,
                        help=f"NMS IoU threshold (default: {DEFAULT_IOU}).")
    parser.add_argument("--device", default="",
                        help="Device: 'cpu', '0', 'mps' … (default: auto).")
    # Output
    parser.add_argument("--output", "-o", default="./inference_output",
                        help="Directory for saved outputs (default: ./inference_output).")
    parser.add_argument("--save-image", action="store_true",
                        help="Save annotated image(s) to --output.")
    parser.add_argument("--save-json",  action="store_true",
                        help="Save detection JSON alongside each image.")
    parser.add_argument("--json-stdout", action="store_true",
                        help="Print full JSON results to stdout (one object per line).")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip drawing boxes (faster when only JSON is needed).")

    args = parser.parse_args(argv)

    # ── Collect sources ───────────────────────────────────────────────────────
    source_path = Path(args.source)
    image_exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if source_path.is_dir():
        sources = sorted(p for p in source_path.rglob("*") if p.suffix.lower() in image_exts)
    elif source_path.is_file():
        sources = [source_path]
    else:
        sys.exit(f"[ERROR] --source not found: {source_path}")

    if not sources:
        sys.exit(f"[ERROR] No image files found in: {source_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    detector  = IDCardDetector(args.model, conf=args.conf, iou=args.iou, device=args.device)
    out_dir   = Path(args.output)
    annotate  = not args.no_annotate

    print(f"\n[i] Processing {len(sources)} image(s) …\n")

    total_det = 0
    for img_path in sources:
        result = detector.predict(img_path, annotate=annotate)
        total_det += result.num_detections

        # ── Console summary ───────────────────────────────────────────────────
        print(f"  {img_path.name:<40}  {result.num_detections:>2} detections  "
              f"({result.inference_ms:.0f} ms)")
        for det in result.detections:
            print(f"      [{det.class_id:>2}] {det.class_name:<18}  "
                  f"conf={det.confidence:.3f}  "
                  f"box=({det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f})")

        # ── Save annotated image ──────────────────────────────────────────────
        if args.save_image and annotate:
            out_img = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"
            detector.save_annotated(result, out_img)
            print(f"      → saved image: {out_img}")

        # ── Save JSON ─────────────────────────────────────────────────────────
        if args.save_json:
            out_json = out_dir / f"{img_path.stem}_detections.json"
            detector.save_json(result, out_json)
            print(f"      → saved JSON : {out_json}")

        # ── JSON stdout ───────────────────────────────────────────────────────
        if args.json_stdout:
            print(result.to_json())

    print(f"\n[✓] Done — {len(sources)} image(s), {total_det} total detections.")
    if args.save_image or args.save_json:
        print(f"    Output directory: {out_dir.resolve()}")


# ===========================================================================
# SECTION 4 – Streamlit UI
# ===========================================================================

def run_streamlit() -> None:
    """
    Full Streamlit web application.  Launch with:
        streamlit run inference.py
    """
    import streamlit as st
    from PIL import ImageDraw as PilDraw, ImageFont as PilFont

    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title = "ប័ណ្ណសំគាល់ · Khmer ID Detector",
        page_icon  = "🪪",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Manrope:wght@300;500;700;800&display=swap');

    :root {
        --bg:        #0d0f14;
        --surface:   #151820;
        --border:    #252a35;
        --accent:    #00e5ff;
        --accent2:   #7c3aed;
        --text:      #e8eaf0;
        --muted:     #6b7280;
        --success:   #10b981;
        --warn:      #f59e0b;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        font-family: 'Manrope', sans-serif;
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    h1 { font-size: 1.7rem !important; font-weight: 800 !important;
         letter-spacing: -0.03em !important; color: var(--text) !important; }
    h2 { font-size: 1.1rem !important; font-weight: 700 !important;
         color: var(--accent) !important; text-transform: uppercase;
         letter-spacing: 0.08em !important; }
    h3 { font-size: 0.9rem !important; font-weight: 600 !important;
         color: var(--muted) !important; text-transform: uppercase;
         letter-spacing: 0.06em !important; }
    .metric-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem; font-weight: 600; color: var(--accent);
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
        letter-spacing: 0.08em; margin-top: 4px;
    }
    .det-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px; margin-bottom: 6px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; transition: border-color 0.2s;
    }
    .det-row:hover { border-color: var(--accent); }
    .det-badge {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
        font-weight: 600; padding: 3px 8px; border-radius: 4px;
        white-space: nowrap;
    }
    .det-name { font-weight: 600; font-size: 0.88rem; flex: 1; }
    .det-conf {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
        color: var(--muted);
    }
    .det-box {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: var(--muted);
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
        color: white !important; border: none !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 700 !important; letter-spacing: 0.03em !important;
        padding: 10px 28px !important; border-radius: 8px !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
    }
    .stSlider > div { color: var(--text) !important; }
    hr { border-color: var(--border) !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar – model config ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🪪 Khmer ID Detector")
        st.markdown("---")

        model_path = st.text_input(
            "Model weights path",
            value=DEFAULT_MODEL,
            help="Path to best.pt from a training run.",
        )
        conf_thresh = st.slider(
            "Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
            help="Lower = more detections (including weak ones).",
        )
        iou_thresh = st.slider(
            "IoU NMS threshold", 0.1, 0.9, DEFAULT_IOU, 0.05,
            help="Lower = more aggressive duplicate suppression.",
        )
        device = st.selectbox(
            "Device", ["auto", "cpu", "0", "mps"],
            help="'0' = first CUDA GPU, 'mps' = Apple Silicon.",
        )
        show_class_filter = st.multiselect(
            "Show only classes",
            options=CLASS_NAMES,
            default=[],
            help="Leave empty to show all classes.",
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#4b5563;line-height:1.6'>"
            "YOLO11n · 12-class detection<br>"
            "Khmer National ID cards<br>"
            "Synthetic training data"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Load / cache detector ─────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading model weights…")
    def load_detector(path: str, conf: float, iou: float, dev: str) -> IDCardDetector:
        return IDCardDetector(
            model_path = path,
            conf       = conf,
            iou        = iou,
            device     = "" if dev == "auto" else dev,
        )

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown("## ប័ណ្ណសំគាល់ · Khmer ID Card Detector")
        st.markdown(
            "<p style='color:#6b7280;font-size:0.9rem;margin-top:-8px'>"
            "Upload an ID card image to detect and localise all 12 text fields."
            "</p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            "<div style='margin-top:14px;text-align:right'>"
            "<span style='background:#1e293b;border:1px solid #334155;"
            "border-radius:6px;padding:4px 10px;font-size:0.72rem;"
            "font-family:IBM Plex Mono;color:#94a3b8'>YOLO11n</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop a card image here  ·  JPG / PNG / BMP / WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="visible",
    )

    if uploaded is None:
        # Welcome state
        st.markdown(
            "<div style='text-align:center;padding:60px 0;color:#374151'>"
            "<div style='font-size:3rem'>🪪</div>"
            "<div style='margin-top:12px;font-size:0.95rem'>Upload an ID card image to begin</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        detector = load_detector(model_path, conf_thresh, iou_thresh, device)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection…"):
        result = detector.predict(img_bgr, annotate=True)

    # Apply class filter
    shown = (
        [d for d in result.detections if d.class_name in show_class_filter]
        if show_class_filter else result.detections
    )

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    def metric_card(col, value, label):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='value'>{value}</div>"
            f"<div class='label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    metric_card(m1, result.num_detections, "Fields detected")
    metric_card(m2, f"{result.inference_ms:.0f} ms", "Inference time")
    metric_card(m3, f"{result.image_width}×{result.image_height}", "Image size")
    avg_conf = (
        round(sum(d.confidence for d in result.detections) / result.num_detections, 3)
        if result.detections else 0.0
    )
    metric_card(m4, f"{avg_conf:.3f}", "Avg confidence")

    st.markdown("")

    # ── Two-column layout: annotated image | detections list ─────────────────
    col_img, col_dets = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown("#### Annotated Image")
        # Redraw only the filtered detections if a filter is active
        if show_class_filter:
            img_filtered = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img_disp = IDCardDetector._draw_boxes(img_filtered, shown)
        else:
            img_disp = result.annotated_image

        st.image(
            cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB),
            use_column_width=True,
            caption=f"{uploaded.name}  ·  {result.inference_ms:.0f} ms",
        )

        # Download button for annotated image
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
        st.download_button(
            label      = "⬇  Download annotated image",
            data       = buf.getvalue(),
            file_name  = f"{Path(uploaded.name).stem}_annotated.jpg",
            mime       = "image/jpeg",
        )

    with col_dets:
        st.markdown(f"#### Detections ({len(shown)})")

        if not shown:
            st.markdown(
                "<div style='color:#6b7280;font-size:0.9rem;padding:20px 0'>"
                "No detections above the confidence threshold.</div>",
                unsafe_allow_html=True,
            )
        else:
            for det in shown:
                color_rgb = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]
                # BGR → CSS hex
                hex_col = "#{:02x}{:02x}{:02x}".format(
                    color_rgb[2], color_rgb[1], color_rgb[0]
                )
                st.markdown(
                    f"<div class='det-row'>"
                    f"<span class='det-badge' style='background:{hex_col}22;"
                    f"color:{hex_col};border:1px solid {hex_col}55'>"
                    f"cls {det.class_id:02d}</span>"
                    f"<span class='det-name'>{det.class_name}</span>"
                    f"<span class='det-conf'>{det.confidence:.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── JSON export ───────────────────────────────────────────────────────
        st.markdown("#### Export JSON")
        json_str = result.to_json()
        st.download_button(
            label     = "⬇  Download detection JSON",
            data      = json_str.encode("utf-8"),
            file_name = f"{Path(uploaded.name).stem}_detections.json",
            mime      = "application/json",
        )
        with st.expander("Preview JSON", expanded=False):
            st.code(json_str, language="json")

        # ── Raw detections table ──────────────────────────────────────────────
        if shown:
            import pandas as pd
            st.markdown("#### Bounding Boxes (pixels)")
            df = pd.DataFrame([
                {
                    "class": det.class_name,
                    "conf":  round(det.confidence, 3),
                    "x1":    int(det.x1), "y1": int(det.y1),
                    "x2":    int(det.x2), "y2": int(det.y2),
                    "w":     int(det.x2 - det.x1),
                    "h":     int(det.y2 - det.y1),
                }
                for det in shown
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ===========================================================================
# SECTION 5 – Entry point dispatcher
# ===========================================================================

def _is_streamlit() -> bool:
    """Return True when this file is being executed by the Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    # Launched via: streamlit run inference.py
    run_streamlit()
elif __name__ == "__main__":
    # Launched via: python inference.py --source ...
    run_cli()
# else: imported as a module → IDCardDetector is available for use
