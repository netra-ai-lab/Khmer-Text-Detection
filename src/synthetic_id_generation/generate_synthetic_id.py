"""
Synthetic Khmer National ID Card Generator for YOLO Text Detection
===================================================================
Generates synthetic ID card images with YOLO-format multi-class annotations.

Classes:
    0: id_number        – Top-right ID number
    1: name_kh          – Name in Khmer script
    2: name_en          – Name in Latin/English
    3: dob_sex_height   – Date of Birth, Sex, Height
    4: pob              – Place of Birth
    5: address_1        – Address line 1
    6: address_2        – Address line 2
    7: validity         – Issue / Expiry dates
    8: features         – Distinguishing features
    9: mrz_1            – MRZ line 1
    10: mrz_2           – MRZ line 2
    11: mrz_3           – MRZ line 3

Directory layout expected:
    ./backgrounds/      – background images (.jpg / .png)
    ./fonts/khmer/      – Khmer .ttf / .otf fonts
    ./fonts/english/    – Latin .ttf / .otf fonts
    ./fonts/mrz/        – OCR-B style .ttf / .otf fonts
    ./output/images/    – generated images  (created automatically)
    ./output/labels/    – generated labels  (created automatically)

Usage:
    python generate_synthetic_id.py --count 200 --output ./output
"""

import os
import sys
import random
import argparse
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Card dimensions (pixels)
# ---------------------------------------------------------------------------
CARD_W = 1000
CARD_H = 630

# ---------------------------------------------------------------------------
# Photo placeholder (left side of card)
# ---------------------------------------------------------------------------
PHOTO_X1 = 30
PHOTO_Y1 = 80
PHOTO_X2 = 280
PHOTO_Y2 = 400

# ---------------------------------------------------------------------------
# Strict layout constants
# ---------------------------------------------------------------------------
TEXT_LEFT  = 310   # x-start for all Khmer/English text fields
TEXT_RIGHT = 960   # hard right boundary - no text may exceed this
MRZ_LEFT   = 40    # MRZ x-start
MRZ_RIGHT  = 970   # MRZ right boundary

# ---------------------------------------------------------------------------
# Hardcoded Y-coordinates (top-left anchor) for every field.
# Fixed so zones never overlap regardless of font or text length.
# ---------------------------------------------------------------------------
FIELD_Y = {
    "id_number":       50,   # right-aligned near X=750
    "name_kh":        110,
    "name_en":        155,
    "dob_sex_height": 200,
    "pob":            245,
    "address_1":      290,
    "address_2":      335,
    "validity":       380,
    "features":       425,
    # ---- gap ----
    "mrz_1":          490,
    "mrz_2":          530,
    "mrz_3":          570,
}

# ---------------------------------------------------------------------------
# Default (maximum) font sizes. draw_text_autofit() may shrink these.
# ---------------------------------------------------------------------------
FONT_SIZE_KH  = 25   # Khmer / English fields
FONT_SIZE_EN  = 25
FONT_SIZE_ID  = 23   # ID number
FONT_SIZE_MRZ = 24   # MRZ monospace
FONT_SIZE_MIN = 10   # absolute floor - never shrink below this

# ---------------------------------------------------------------------------
# Helper – Western digit → Khmer numeral
# ---------------------------------------------------------------------------
def to_khmer_num(text: str) -> str:
    """Replace every ASCII digit in *text* with its Khmer-script equivalent."""
    _MAP = {
        "0": "០", "1": "១", "2": "២", "3": "៣", "4": "៤",
        "5": "៥", "6": "៦", "7": "៧", "8": "៨", "9": "៩",
    }
    return "".join(_MAP.get(ch, ch) for ch in text)


# ---------------------------------------------------------------------------
# Dummy data pools (Extended for high variance)
# ---------------------------------------------------------------------------
DUMMY_KHMER_NAMES =[
    "សុខ សុភាព", "ហុង វណ្ណា", "ជាវ កូឡាវីន", "នុត សុផល",
    "កែវ រតនា", "ស៊ឹម សុវណ្ណារិទ្ធ", "ឈួន ស្រីរ័ត្ន", "មាស វឌ្ឍនៈ",
    "អ៊ុក បូរី", "លឹម លីហ្សា", "ផាន់ ធីតា", "រស់ សេរីវុធ",
    "ឌិន ចាន់ថន", "វ៉ាន់ ដារ៉ា", "សោម ស្រីនាង", "ព្រំ សុវណ្ណ"
]

DUMMY_ENGLISH_NAMES =[
    "SOK SOPHEAK", "HONG VANNA", "CHEAV COLAWIN", "NUT SOPHAL",
    "KEO RATANA", "SIM SOVANNARITH", "CHHUON SREYROTH", "MEAS VATHANAK",
    "OUK BOREY", "LIM LYZA", "PHAN THIDA", "ROS SEREYVUTH",
    "DIN CHANTHON", "VAN DARA", "SOM SREYNEANG", "PROM SOVANN"
]

# Dates use dot-separator (DD.MM.YYYY); to_khmer_num() converts digits at render time
DUMMY_DOBS =[
    "21.06.2009", "12.03.1985", "07.11.1992", "25.06.1978",
    "05.12.1990", "14.02.2001", "30.08.1988", "19.04.1995",
    "22.11.1975", "01.01.1982", "15.05.1998", "08.09.2003"
]

DUMMY_SEXES = ["ប្រុស", "ស្រី"]   # Male / Female in Khmer
DUMMY_HEIGHTS =["148", "150", "155", "160", "162", "165", "168", "170", "172", "175", "178", "180"]

# Varied places of birth spanning multiple provinces and formats
DUMMY_POBS =[
    "សង្កាត់តាខ្មៅ ក្រុងតាខ្មៅ កណ្ដាល",
    "ភូមិ៣ សង្កាត់បឹងព្រលិត ខណ្ឌ៧មករា ភ្នំពេញ",
    "ភូមិវត្តបូព៌ សង្កាត់សាលាកំរើក ក្រុងសៀមរាប សៀមរាប",
    "ភូមិ២ ឃុំត្រពាំងឫស្សី ស្រុកកំពង់ស្វាយ កំពង់ធំ",
    "ភូមិព្រៃវែង ឃុំព្រៃវែង ស្រុកពារាំង ព្រៃវែង",
    "ភូមិកំពង់ឆ្នាំង សង្កាត់កំពង់ឆ្នាំង ក្រុងកំពង់ឆ្នាំង កំពង់ឆ្នាំង",
    "មន្ទីរពេទ្យកាល់ម៉ែត ភ្នំពេញ"
]

# Varied Address lines (House numbers, streets, villages)
DUMMY_ADDR1 =[
    "ភូមិតាខ្មៅ១",
    "ផ្ទះលេខ ១២ ផ្លូវ ២៥៩",
    "ផ្ទះលេខ ៤៥ ផ្លូវ ៣១៥ ភូមិ៣",
    "ក្រុមទី៤ ភូមិវត្តបូព៌",
    "ផ្ទះលេខ ៩ ផ្លូវលំ ភូមិ៣",
    "ភូមិព្រៃវែង",
    "ផ្ទះលេខ ៨៨ ផ្លូវជាតិលេខ ៥"
]

# Varied Communes (Sangkat), Districts (Khan/Krong), and Provinces
DUMMY_ADDR2 =[
    "សង្កាត់តាខ្មៅ ក្រុងតាខ្មៅ កណ្ដាល",
    "សង្កាត់ ទួលទំពូង១ ខណ្ឌ ចំការមន ភ្នំពេញ",
    "សង្កាត់បឹងកក់ទី១ ខណ្ឌទួលគោក ភ្នំពេញ",
    "សង្កាត់សាលាកំរើក ក្រុងសៀមរាប សៀមរាប",
    "ឃុំត្រពាំងឫស្សី ស្រុកកំពង់ស្វាយ កំពង់ធំ",
    "សង្កាត់ចោមចៅទី១ ខណ្ឌពោធិ៍សែនជ័យ ភ្នំពេញ"
]

# Full DD.MM.YYYY issue / expiry dates (usually exactly 10 years apart)
DUMMY_ISSUES =[
    "03.04.2024", "15.08.2018", "10.01.2020",
    "01.06.2015", "12.11.2021", "20.02.2019", "08.09.2023"
]
DUMMY_EXPIRIES =[
    "02.04.2034", "15.08.2028", "10.01.2030",
    "01.06.2025", "12.11.2031", "20.02.2029", "08.09.2033"
]

# Common facial features/scars in Khmer
DUMMY_FEATURES =[
    "គ្មាន",
    "ប្រជ្រុយចំ.៥ស.ម មុខក្រោមចុងដង្ហើមខាងឆ្វេង",
    "ស្លាកស្នាមលើចិញ្ចើមស្តាំ",
    "ស្នាមសម្លាកលើចិញ្ចើមឆ្វេង",
    "ប្រជ្រុយក្រោមភ្នែកស្តាំ",
    "ស្នាមសម្លាកក្រោមកប្រហែល៣ស.ម",
    "ប្រជ្រុយលើបបូរមាត់",
    "គ្មានភិនភាគសម្គាល់"
]

# Static MRZ reference lines (ICAO TD-1, exactly 30 chars per line).
# Added new ones corresponding to the new English names.
DUMMY_MRZ_SETS: list[tuple[str, str, str]] =[
    (
        "IDKHM0214540875<<<<<<<<<<<<<<<",
        "0906216M3408021KHM<<<<<<<<<<<0",
        "CHEAV<<COLAWIN<<<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMSOPHEAK0001<<<<<<<<<<<<<<<",
        "8503124F2803121KHM<<<<<<<<<<<6",
        "SOK<<SOPHEAK<<<<<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMVANNA00023<<<<<<<<<<<<<<<<",
        "9211074F2811071KHM<<<<<<<<<<<2",
        "HONG<<VANNA<<<<<<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMSOPHAL0034<<<<<<<<<<<<<<<<",
        "7806254F2806251KHM<<<<<<<<<<<8",
        "NUT<<SOPHAL<<<<<<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMKEO000145<<<<<<<<<<<<<<<<",
        "9012054M3106011KHM<<<<<<<<<<<8",
        "KEO<<RATANA<<<<<<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMSIM000256<<<<<<<<<<<<<<<<",
        "0102144M3111121KHM<<<<<<<<<<<5",
        "SIM<<SOVANNARITH<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMCHHUON003<<<<<<<<<<<<<<<<",
        "8808304F2902201KHM<<<<<<<<<<<3",
        "CHHUON<<SREYROTH<<<<<<<<<<<<<<",
    ),
    (
        "IDKHMMEAS00045<<<<<<<<<<<<<<<<",
        "9504194M3309081KHM<<<<<<<<<<<9",
        "MEAS<<VATHANAK<<<<<<<<<<<<<<<<",
    )
]

# MRZ character set (kept for any future dynamic generation helpers)
MRZ_CHARS    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
COUNTRY_CODE = "KHM"


# ===========================================================================
# SECTION 1 – Utility helpers
# ===========================================================================

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def bbox_to_yolo(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int = CARD_W, img_h: int = CARD_H,
    pad_x: int = 4, pad_y: int = 2,
) -> tuple[float, float, float, float]:
    """
    Convert pixel bbox (x1,y1,x2,y2) to normalised YOLO format
    (x_center, y_center, width, height) with optional padding.
    All returned values are clamped to [0, 1].
    """
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    return (
        clamp(cx), clamp(cy),
        clamp(w),  clamp(h),
    )


# ===========================================================================
# SECTION 2 – Background loading
# ===========================================================================

def load_backgrounds(bg_dir: str = "./backgrounds") -> list[Path]:
    """
    Scan *bg_dir* for .jpg / .png files and return a list of Paths.
    Falls back to a solid-colour placeholder if the directory is empty/missing.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    bg_path   = Path(bg_dir)
    if bg_path.is_dir():
        files = [p for p in bg_path.iterdir() if p.suffix.lower() in supported]
        if files:
            return files
    print(
        f"[WARNING] No backgrounds found in '{bg_dir}'. "
        "A plain colour will be used instead."
    )
    return []


def get_background(
    bg_files: list[Path],
    target_w: int = CARD_W,
    target_h: int = CARD_H,
) -> Image.Image:
    """
    Pick a random background image, resize + centre-crop it to (target_w × target_h).
    If no files are available, return a plain off-white image.
    """
    if not bg_files:
        colour = (
            random.randint(230, 255),
            random.randint(230, 255),
            random.randint(230, 255),
        )
        return Image.new("RGB", (target_w, target_h), colour)

    path  = random.choice(bg_files)
    img   = Image.open(path).convert("RGB")
    iw, ih = img.size

    # Scale so the shorter dimension covers the target
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)

    # Centre crop
    left = (nw - target_w) // 2
    top  = (nh - target_h) // 2
    img  = img.crop((left, top, left + target_w, top + target_h))
    return img


# ===========================================================================
# SECTION 3 – Font loading
# ===========================================================================

def scan_fonts(font_dir: str) -> list[Path]:
    """Return all .ttf / .otf files found (recursively) inside *font_dir*."""
    supported = {".ttf", ".otf"}
    root = Path(font_dir)
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in supported]


def load_font_pool(
    khmer_dir:  str = "./fonts/khmer",
    english_dir: str = "./fonts/english",
    mrz_dir:    str = "./fonts/mrz",
) -> dict[str, list[Path]]:
    """
    Scan the three font directories and return a dict with keys
    'khmer', 'english', 'mrz' mapping to lists of font Paths.
    Missing directories print a warning and use Pillow's built-in default.
    """
    pools: dict[str, list[Path]] = {}
    for key, directory in [
        ("khmer",   khmer_dir),
        ("english", english_dir),
        ("mrz",     mrz_dir),
    ]:
        files = scan_fonts(directory)
        if not files:
            print(
                f"[WARNING] No fonts found in '{directory}'. "
                f"Pillow default will be used for '{key}' fields."
            )
        pools[key] = files
    return pools


def pick_font(
    font_files: list[Path],
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a random font from *font_files* at *size*.
    Falls back to Pillow's built-in bitmap font if the list is empty.
    """
    if not font_files:
        # Pillow default – no size argument accepted
        return ImageFont.load_default()
    path = random.choice(font_files)
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError as exc:
        print(f"[WARNING] Could not load font '{path}': {exc}. Using default.")
        return ImageFont.load_default()


# ===========================================================================
# SECTION 4 – Dummy data generation
# ===========================================================================

def _random_mrz_str(length: int) -> str:
    return "".join(random.choices(MRZ_CHARS, k=length))




def generate_field_texts(name_idx: int | None = None) -> dict[str, str]:
    """
    Build a dictionary mapping field names → display strings for one card.

    All formatting rules are applied here to match a real Cambodian ID layout:
      – Khmer prefix labels are prepended to each field.
      – Dates use dot-separators (DD.MM.YYYY).
      – Dates and heights are rendered in Khmer numerals via to_khmer_num().
      – Sex is in Khmer (ប្រុស / ស្រឹ).
      – MRZ is selected from a static reference pool (ASCII only, MRZ font).

    Pass *name_idx* to deterministically pick a name (useful for testing).
    """
    idx = name_idx if name_idx is not None else random.randrange(len(DUMMY_ENGLISH_NAMES))
    idx = idx % len(DUMMY_ENGLISH_NAMES)

    name_kh = DUMMY_KHMER_NAMES[idx % len(DUMMY_KHMER_NAMES)]
    name_en = DUMMY_ENGLISH_NAMES[idx]

    # ---- Field text construction with real ID formatting ----

    # Class 0 – ID number: plain Western digits (as on the real card)
    id_number = "0" + "".join([str(random.randint(0, 9)) for _ in range(8)])

    # Class 1 – Name in Khmer: prefix + name
    name_kh_text = f"គោត្តនាមនិងនាម៖ {name_kh}"

    # Class 2 – Name in English/Latin: plain
    name_en_text = name_en

    # Class 3 – DOB / Sex / Height (all numbers → Khmer numerals)
    dob    = to_khmer_num(random.choice(DUMMY_DOBS))
    sex    = random.choice(DUMMY_SEXES)
    height = to_khmer_num(random.choice(DUMMY_HEIGHTS))
    dob_line = (
        f"ថ្ងៃខែឆ្នាំកំណើត៖ {dob}"
        f" ភេទ៖ {sex}"
        f" កម្ពស់៖ {height} ស.ម"
    )

    # Class 4 – Place of birth
    pob_text = f"ទីកន្លែងកំណើត៖ {random.choice(DUMMY_POBS)}"

    # Class 5 – Address line 1
    addr1_text = f"អាសយដ្ឋាន៖ {random.choice(DUMMY_ADDR1)}"

    # Class 6 – Address line 2 (continuation, no prefix)
    addr2_text = random.choice(DUMMY_ADDR2)

    # Class 7 – Validity: issue & expiry dates (Khmer numerals)
    issue  = to_khmer_num(random.choice(DUMMY_ISSUES))
    expiry = to_khmer_num(random.choice(DUMMY_EXPIRIES))
    validity_text = f"សុពលភាព៖ {issue} ដល់ {expiry}"

    # Class 8 – Distinguishing features
    features_text = f"និងភិនភាគ៖ {random.choice(DUMMY_FEATURES)}"

    # Classes 9–11 – MRZ lines (static reference pool, ASCII only)
    mrz1, mrz2, mrz3 = random.choice(DUMMY_MRZ_SETS)

    return {
        "id_number":      id_number,
        "name_kh":        name_kh_text,
        "name_en":        name_en_text,
        "dob_sex_height": dob_line,
        "pob":            pob_text,
        "address_1":      addr1_text,
        "address_2":      addr2_text,
        "validity":       validity_text,
        "features":       features_text,
        "mrz_1":          mrz1,
        "mrz_2":          mrz2,
        "mrz_3":          mrz3,
    }

# ===========================================================================
# SECTION 5 - Text rendering + bounding-box extraction
# ===========================================================================

# ---------------------------------------------------------------------------
# Per-field rendering metadata
#   class_id  : YOLO class index (0-11)
#   font_key  : which font pool to use ('khmer' | 'english' | 'mrz')
#   base_size : default (maximum) font size; autofit may shrink this
#   x_start   : left-edge pixel for this field
#   max_x     : rightmost pixel the text is allowed to reach
# ---------------------------------------------------------------------------
FIELD_META: dict[str, dict] = {
"id_number":       dict(class_id=0,  font_key="english", base_size=FONT_SIZE_ID,  x_start=720, y_start=120, max_x=TEXT_RIGHT),
    "name_kh":         dict(class_id=1,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "name_en":         dict(class_id=2,  font_key="english", base_size=FONT_SIZE_EN,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "dob_sex_height":  dict(class_id=3,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "pob":             dict(class_id=4,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "address_1":       dict(class_id=5,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "address_2":       dict(class_id=6,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "validity":        dict(class_id=7,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "features":        dict(class_id=8,  font_key="khmer",   base_size=FONT_SIZE_KH,  x_start=TEXT_LEFT, max_x=TEXT_RIGHT),
    "mrz_1":           dict(class_id=9,  font_key="mrz",     base_size=FONT_SIZE_MRZ, x_start=MRZ_LEFT,  max_x=MRZ_RIGHT),
    "mrz_2":           dict(class_id=10, font_key="mrz",     base_size=FONT_SIZE_MRZ, x_start=MRZ_LEFT,  max_x=MRZ_RIGHT),
    "mrz_3":           dict(class_id=11, font_key="mrz",     base_size=FONT_SIZE_MRZ, x_start=MRZ_LEFT,  max_x=MRZ_RIGHT),
}


def _text_color() -> tuple[int, int, int]:
    # Return a slightly randomised dark ink colour for realism.
    v = random.randint(10, 60)
    return (v, v, v)


def draw_text_autofit(
    draw:       "ImageDraw.ImageDraw",
    font_files: list,
    text:       str,
    x_start:    int,
    y_start:    int,
    base_size:  int,
    max_x:      int,
    colour:     tuple[int, int, int],
    min_size:   int = FONT_SIZE_MIN,
) -> tuple[int, int, int, int]:
    # Render *text* at (x_start, y_start), shrinking font size until the text
    # fits within *max_x*.  Returns the tight pixel bbox (x1, y1, x2, y2).
    #
    # Algorithm:
    #   1. Pick one font file randomly (fixed across size iterations).
    #   2. Load at base_size, measure with draw.textbbox(anchor='lt').
    #   3. Decrement size by 1 and reload until bx2 <= max_x OR size == min_size.
    #   4. Draw with the final font and return bbox clamped to card bounds.
    #
    # Guarantees:
    #   - No text ever overflows the right edge.
    #   - YOLO boxes map to the *actually rendered* glyph extent.

    chosen_file = random.choice(font_files) if font_files else None

    def _load(sz):
        if chosen_file is None:
            return ImageFont.load_default()
        try:
            return ImageFont.truetype(str(chosen_file), size=sz)
        except OSError:
            return ImageFont.load_default()

    def _measure(fnt):
        try:
            return draw.textbbox((x_start, y_start), text, font=fnt, anchor="lt")
        except TypeError:
            return draw.textbbox((x_start, y_start), text, font=fnt)

    size = base_size
    fnt  = _load(size)
    bx1, by1, bx2, by2 = _measure(fnt)

    # Shrink loop - reduce size by 1 each iteration until text fits
    while bx2 > max_x and size > min_size:
        size -= 1
        fnt   = _load(size)
        bx1, by1, bx2, by2 = _measure(fnt)

    # Draw with the winning font
    try:
        draw.text((x_start, y_start), text, font=fnt, fill=colour, anchor="lt")
    except TypeError:
        draw.text((x_start, y_start), text, font=fnt, fill=colour)

    # Clamp bbox to card boundaries (handles rare Khmer descender overflows)
    bx1 = max(0,      bx1)
    by1 = max(0,      by1)
    bx2 = min(CARD_W, bx2)
    by2 = min(CARD_H, by2)

    return bx1, by1, bx2, by2


def render_text_and_collect_bboxes(
    card:        "Image.Image",
    draw:        "ImageDraw.ImageDraw",
    field_texts: dict[str, str],
    font_pool:   dict[str, list],
) -> list[tuple[int, float, float, float, float]]:
    # Draw all text fields onto *card* and return YOLO annotation tuples.
    #
    # Every field is routed through draw_text_autofit() which:
    #   - Enforces hard X boundaries via per-field max_x (no horizontal overflow).
    #   - Uses hardcoded Y anchors from FIELD_Y (no vertical overlap / bleed).
    #   - Returns the true rendered bbox for pixel-accurate YOLO boxes.
    #
    # font_pool: dict with keys 'khmer', 'english', 'mrz' -> list[Path].
    # Passed directly so autofit can reload fonts at any point size.

    annotations: list[tuple[int, float, float, float, float]] = []

    for field_name, text in field_texts.items():
        meta     = FIELD_META[field_name]
        class_id = meta["class_id"]
        x_start  = meta["x_start"]
        max_x    = meta["max_x"]
        base_sz  = meta["base_size"]
        files    = font_pool.get(meta["font_key"], [])
        y_start  = FIELD_Y[field_name]
        colour   = _text_color()

        bx1, by1, bx2, by2 = draw_text_autofit(
            draw       = draw,
            font_files = files,
            text       = text,
            x_start    = x_start,
            y_start    = y_start,
            base_size  = base_sz,
            max_x      = max_x,
            colour     = colour,
        )

        # Convert pixel bbox to normalised YOLO (adds small padding)
        cx, cy, w, h = bbox_to_yolo(bx1, by1, bx2, by2)

        # Discard degenerate boxes (empty string, font fallback issue, etc.)
        if w > 0.005 and h > 0.005:
            annotations.append((class_id, cx, cy, w, h))

    return annotations


# ===========================================================================
# SECTION 6 – Card overlay decorations
# ===========================================================================

OVERLAY_ALPHA = 180   # 0 = transparent, 255 = opaque semi-transparent card layer

def draw_card_overlay(card: Image.Image) -> Image.Image:
    """
    Draw a semi-transparent white card overlay to simulate the ID card's
    printed surface on top of the background.  Also renders the photo
    placeholder rectangle and some decorative horizontal lines.
    Returns a composite RGB image.
    """
    overlay = Image.new("RGBA", card.size, (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    # Card body (slight off-white tint)
    card_colour = (
        random.randint(245, 255),
        random.randint(245, 255),
        random.randint(245, 255),
        OVERLAY_ALPHA,
    )
    odraw.rectangle([10, 10, CARD_W - 10, CARD_H - 10], fill=card_colour)

    # Photo placeholder
    photo_fill = (random.randint(180, 210),) * 3 + (220,)
    odraw.rectangle([PHOTO_X1, PHOTO_Y1, PHOTO_X2, PHOTO_Y2], fill=photo_fill, outline=(100, 100, 100, 200), width=2)

    # Top colour bar (flag-inspired accent)
    bar_colour = (0, 56, 168, 200)  # Cambodian blue
    odraw.rectangle([10, 10, CARD_W - 10, 55], fill=bar_colour)

    # MRZ separator line
    mrz_y = CARD_H - 130
    odraw.line([(10, mrz_y), (CARD_W - 10, mrz_y)], fill=(120, 120, 120, 180), width=1)

    # Composite
    return Image.alpha_composite(card.convert("RGBA"), overlay).convert("RGB")


# ===========================================================================
# SECTION 7 – Augmentation pipeline (spatial distortions are FORBIDDEN)
# ===========================================================================

def build_augmentation_pipeline() -> A.Compose:
    """
    Build an albumentations pipeline with ONLY photometric augmentations.
    No geometric/spatial transforms are included to preserve bbox validity.
    """
    return A.Compose([
        # Brightness / contrast
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        # Hue / saturation / value
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        # CLAHE – local contrast enhancement (simulates uneven lighting)
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        # Gaussian noise (sensor / scanner grain)
        # albumentations 2.x: std_range replaces var_limit
        A.GaussNoise(std_range=(0.02, 0.10), p=0.5),
        # Slight blur (defocus / motion)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.35),
        # Shadow / illumination gradient
        # albumentations 2.x: num_shadows_limit replaces num_shadows_lower/upper
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.3,
        ),
        # Slight colour shift
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        # Overall image compression artefact simulation
        # albumentations 2.x: quality_range replaces quality_lower/quality_upper
        A.ImageCompression(quality_range=(70, 100), p=0.25),
    ])


def apply_augmentations(
    image_np: np.ndarray,
    pipeline: A.Compose,
) -> np.ndarray:
    """Apply the augmentation pipeline to a numpy (H,W,3) BGR or RGB image."""
    result = pipeline(image=image_np)
    return result["image"]


# ===========================================================================
# SECTION 8 – Full card generation
# ===========================================================================

def generate_card(
    bg_files:     list[Path],
    font_pool:    dict[str, list[Path]],
    aug_pipeline: A.Compose,
    name_idx:     int | None = None,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    """
    Generate one synthetic ID card image together with its YOLO annotations.

    Returns:
        (PIL Image (RGB), list of (class_id, cx, cy, w, h) tuples)
    """
    # 1. Background
    card = get_background(bg_files, CARD_W, CARD_H)

    # 2. Card overlay + decorations
    card = draw_card_overlay(card)

    # 3. font_pool (raw Path lists) is passed directly to render_text_and_collect_bboxes
    #    so draw_text_autofit() can reload fonts at any size during shrink iterations.

    # 4. Generate field texts
    field_texts = generate_field_texts(name_idx)

    # 5. Render text + collect bounding boxes
    draw = ImageDraw.Draw(card)
    annotations = render_text_and_collect_bboxes(card, draw, field_texts, font_pool)

    # 6. Augment (photometric only)
    card_np = np.array(card)   # PIL -> numpy RGB
    card_np = apply_augmentations(card_np, aug_pipeline)
    card    = Image.fromarray(card_np)

    return card, annotations


# ===========================================================================
# SECTION 9 – Output writer
# ===========================================================================

def write_outputs(
    card:        Image.Image,
    annotations: list[tuple[int, float, float, float, float]],
    stem:        str,
    img_dir:     Path,
    lbl_dir:     Path,
) -> None:
    """
    Save the card image as JPEG and write its YOLO label file.

    Args:
        card:        PIL Image
        annotations: list of (class_id, cx, cy, w, h)
        stem:        filename stem (no extension)
        img_dir:     output directory for images
        lbl_dir:     output directory for labels
    """
    # Save image
    img_path = img_dir / f"{stem}.jpg"
    card.save(str(img_path), format="JPEG", quality=92)

    # Write YOLO label
    lbl_path = lbl_dir / f"{stem}.txt"
    with open(lbl_path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in annotations:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ===========================================================================
# SECTION 10 – dataset.yaml generator
# ===========================================================================

YAML_TEMPLATE = """\
# YOLO dataset configuration – Khmer National ID card detection
path: {root}
train: images/train
val:   images/val

nc: 12
names:
  0: id_number
  1: name_kh
  2: name_en
  3: dob_sex_height
  4: pob
  5: address_1
  6: address_2
  7: validity
  8: features
  9: mrz_1
  10: mrz_2
  11: mrz_3
"""


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(YAML_TEMPLATE.format(root=str(output_dir.resolve())))
    print(f"[✓] dataset.yaml written → {yaml_path}")


# ===========================================================================
# SECTION 11 – CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Khmer National ID generator for YOLO detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Directory layout expected:
              ./backgrounds/         – background images
              ./fonts/khmer/         – Khmer TTF/OTF fonts
              ./fonts/english/       – English/Latin TTF/OTF fonts
              ./fonts/mrz/           – OCR-B or monospace MRZ fonts
        """),
    )
    parser.add_argument(
        "--count", "-n",
        type=int, default=100,
        help="Number of synthetic images to generate (default: 100).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="./output",
        help="Root output directory (default: ./output).",
    )
    parser.add_argument(
        "--bg-dir",
        type=str, default="./backgrounds",
        help="Directory containing background images.",
    )
    parser.add_argument(
        "--fonts-khmer",
        type=str, default="./fonts/khmer",
        help="Directory containing Khmer fonts.",
    )
    parser.add_argument(
        "--fonts-english",
        type=str, default="./fonts/english",
        help="Directory containing English/Latin fonts.",
    )
    parser.add_argument(
        "--fonts-mrz",
        type=str, default="./fonts/mrz",
        help="Directory containing MRZ (OCR-B) fonts.",
    )
    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--split",
        type=float, default=0.85,
        help="Fraction of images to place in train split (rest go to val). Default: 0.85.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[i] Random seed set to {args.seed}")

    # ---- Prepare output directories ----
    root = Path(args.output)
    splits = ["train", "val"]
    dirs: dict[str, dict[str, Path]] = {}
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = {"images": img_dir, "labels": lbl_dir}

    # ---- Load resources ----
    bg_files  = load_backgrounds(args.bg_dir)
    font_pool = load_font_pool(args.fonts_khmer, args.fonts_english, args.fonts_mrz)
    aug       = build_augmentation_pipeline()

    print(f"\n[i] Generating {args.count} synthetic ID cards …\n")

    n_train = int(args.count * args.split)

    for i in range(args.count):
        split     = "train" if i < n_train else "val"
        stem      = f"kh_id_{i:06d}"
        img_dir   = dirs[split]["images"]
        lbl_dir   = dirs[split]["labels"]

        try:
            card, annotations = generate_card(bg_files, font_pool, aug)
            write_outputs(card, annotations, stem, img_dir, lbl_dir)
        except Exception as exc:
            print(f"[ERROR] Failed on card {i}: {exc}")
            continue

        if (i + 1) % 10 == 0 or (i + 1) == args.count:
            print(f"  [{i+1:>5}/{args.count}] {split}/{stem}.jpg  "
                  f"({len(annotations)} boxes)")

    # ---- dataset.yaml ----
    write_dataset_yaml(root)

    print(
        f"\n[✓] Done!  {args.count} cards generated.\n"
        f"    Train: {n_train}  |  Val: {args.count - n_train}\n"
        f"    Output root: {root.resolve()}\n"
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()