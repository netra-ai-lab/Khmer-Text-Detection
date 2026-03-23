"""
generate_doc_lines.py
=====================
Synthetic Cambodian Official Document Generator
for YOLO Text-Line Detection

Generates portrait-style images that mimic Cambodian ministry press releases,
formal letters, and government documents — then exports tight per-line bounding
boxes in standard YOLO format (single class 0 = text line).

Directory layout expected / created:
    ./fonts/                     ← .ttf / .otf font files (any depth)
    ./texts/khmer_corpus.txt     ← Khmer text corpus (auto-created if missing)
    ./datasets/images/train/     ← output JPEG images        (auto-created)
    ./datasets/labels/train/     ← output YOLO .txt labels   (auto-created)

Usage:
    python generate_doc_lines.py --count 200
    python generate_doc_lines.py --count 500 --output ./datasets --seed 42
"""

import argparse
import re
import random
import textwrap
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# SECTION 0 – Fallback corpus
#   Written to ./texts/khmer_corpus.txt the first time if the file is absent.
#   Extend or replace with a real corpus for higher-quality training data.
# ============================================================================

FALLBACK_CORPUS = """\
រាជរដ្ឋាភិបាលកម្ពុជាបានប្រកាសពីការអនុម័តលើគម្រោងអភិវឌ្ឍន៍ហេដ្ឋារចនាសម្ព័ន្ធថ្មីនៅក្នុងខេត្តព្រះសីហនុ។
ក្រសួងសេដ្ឋកិច្ចនិងហិរញ្ញវត្ថុបានរៀបចំសេចក្ដីព្រាងច្បាប់ស្ដីពីការគ្រប់គ្រងថវិកាជាតិប្រចាំឆ្នាំ។
ការប្រជុំពិភាក្សាអំពីការអភិវឌ្ឍន៍ជនបទត្រូវបានរៀបចំឡើងនៅទីស្ដីការគណៈរដ្ឋមន្ត្រី។
ក្រសួងអប់រំ យុវជន និងកីឡា បានប្រកាសពីការពង្រីកបណ្ដាញសាលារៀននៅតំបន់ជនបទ។
ការអនុវត្តន៍នយោបាយការពារបរិស្ថានត្រូវបានពង្រឹងនៅក្នុងតំបន់ឆ្នេរសមុទ្រ។
ក្រសួងសុខាភិបាលបានចេញសេចក្ដីណែនាំស្ដីពីការការពារជំងឺឆ្លងនៅរដូវវស្សា។
ប្រទេសកម្ពុជាកំពុងអនុវត្តន៍ផែនការយុទ្ធសាស្ត្រអភិវឌ្ឍន៍ជាតិដំណាក់កាលថ្មី។
ក្រុមប្រឹក្សាជាតិអភិវឌ្ឍន៍សេដ្ឋកិច្ចបានពិភាក្សាលើការចូលរួមវិនិយោគបរទេស។
ក្រសួងការបរទេសនិងសហប្រតិបត្តិការអន្តរជាតិបានស្វាគមន៍ការបង្រ្គប់ទំនាក់ទំនងការទូត។
ការប្រើប្រាស់បច្ចេកវិទ្យាព័ត៌មានក្នុងប្រព័ន្ធរដ្ឋបាលសាធារណៈកំពុងត្រូវបានពង្រឹង។
រដ្ឋមន្ត្រីក្រសួងការងារបានជម្រុញឱ្យស្ថាប័នឯកជនបង្កើនជំនួញបំណែចក្រការ។
គម្រោងទំនប់វារីអគ្គិសនីថ្មីនឹងផ្ដល់ថាមពលអគ្គិសនីបន្ថែមដល់ប្រជាពលរដ្ឋ។
ក្រសួងទេសចរណ៍បានប្រកាសពីដំណើរការអភិវឌ្ឍន៍វិស័យទេសចរណ៍នៅសេសសូស្ដីណីស។
ការកសាងផ្លូវវឹរស្ទឹងលើតំបន់ទំនាបភ្លៀងបានផ្ដល់ផលជាវិជ្ជមានដល់កសិករ។
ស្ថាប័នសាធារណៈគ្រប់ជាន់ថ្នាក់ត្រូវតែអនុវត្តន៍ការបិទបាំងព័ត៌មានប្រកបដោយតម្លាភាព។
ក្រុមហ៊ុនវិនិយោគក្នុងស្រុកនិងបរទេសសហការគ្នាក្នុងការអភិវឌ្ឍន៍ឧស្សាហកម្ម។
ជំនួយការបណ្ដុះបណ្ដាលវិជ្ជាជីវៈត្រូវបានផ្ដល់ជូនដល់ក្រុមយុវវ័យនៅទូទាំងប្រទេស។
ក្រសួងរៀបចំដែនដីនគរូបនីយកម្មនិងសំណង់បានអនុម័តច្បាប់ស្ដីពីការគ្រប់គ្រងដី។
ការពង្រឹងតុលាការឯករាជ្យគឺជាអាទិភាពចម្បងក្នុងការកែទម្រង់ច្បាប់។
ក្រសួងព័ត៌មានបានអំពាវនាវឱ្យប្រព័ន្ធផ្សព្វផ្សាយអនុវត្តន៍សីលធម៌វិជ្ជាជីវៈ។
ការបោះឆ្នោតថ្នាក់ក្រោមជាតិបានដំណើរការដោយសន្តិភាពនិងប្រជាធិបតេយ្យ។
ក្រសួងកសិកម្មបានផ្ដល់គ្រឿងយន្តកសិកម្មដល់សហគមន៍ជនបទ។
ស្ថាប័នបណ្ដុះបណ្ដាលបច្ចេកទេសសំខាន់ៗនៅទូទាំងប្រទេសត្រូវបានបង្ករ។
ការសហប្រតិបត្តិការអាស៊ានបានពង្រឹងចំណងរដ្ឋវិជ្ជានៃប្រទេសសហស្ថាបន័រ។
ជំហានសំខាន់ៗក្នុងការការពារមត្រយោគបានបន្ថែមភាពរឹងមាំដល់ប្រព័ន្ធតុលាការ។
ក្រសួងពាណិជ្ជកម្មបានពង្រីករចនាសម្ព័ន្ធទីផ្សារក្នុងស្រុកនិងអន្ដរជាតិ។
ការអប់រំផ្នែកទីជំនិតទស្សនៈជាតិគឺជាគ្រឹះដ៏សំខាន់នៃការបណ្ដុះបណ្ដាលពលរដ្ឋ។
ក្រសួងបរិស្ថានបានបង្ហាញការប្ដេជ្ញាចិត្តក្នុងការការពារព្រៃឈើជាតិ។
ច្បាប់ស្ដីពីការការពារកម្មសិទ្ធិបញ្ញាបានចូលជាធរមានកាន់តែពេញលេញ។
ការធ្វើផែនការទីក្រុងប្រកបដោយចីរភាពគឺជាគោលការណ៍ចម្បងក្នុងការអភិវឌ្ឍ។
ក្រោមការដឹកនាំរបស់ឯកឧត្ដម នាយករដ្ឋមន្ត្រី ការអភិវឌ្ឍជាតិមានឧត្ដគតិ។
ការបណ្ដុះបណ្ដាលធនធានមនុស្សគ្រប់ជំនាញគឺជាការអាទិភាពរបស់រដ្ឋ។
ព្រះរាជអាណាចក្រកម្ពុជាមានទំនាក់ទំនងល្អជាមួយប្រទេសសហការគ្រប់ទិសសមុទ្រ។
ការចូលរួមយ៉ាងសកម្មក្នុងអង្គការអន្ដរជាតិបានពង្រឹងស្ថានភាពការទូតរបស់ប្រទេស។
ក្រសួងអ្នកវប្បធម៌ ល្ខោន ហ្វ្លាំង ។ ការការពារវប្បធម៌ Khmer ។ ការជំរុញការរៀនភាសាបរទេស ។
ការអនុវត្ដច្បាប់ការពារកុមារនិងស្ត្រីឆ្ពោះទៅការបង្ការអំពើហិង្សា ។
ការសិក្សាស្រាវជ្រាវវិទ្យាសាស្ត្របច្ចេកវិទ្យានៅស្ថាប័នឧត្ដមសិក្សា ។
ការទទួលស្គាល់到达ការបំពាក់ប្រព័ន្ធផ្ដល់ព័ត៌មានតាមប្រព័ន្ធអ៊ីនធឺណែត ។
ក្រសួងសង្គមកិច្ចមានកម្មវិធីគ្រប់គ្រងជំនួយសង្គម ។ ការជួយដល់ប្រជាពលរដ្ឋ ។
ការប្ដូរប្រតិភូទៅ​ប្រទេសសហប្រតិបត្ដិការនិងការស្វែងរកសន្ធិសញ្ញាថ្មី ។
ការចូលរួមក្នុងកិច្ចប្រជុំអ.ស.ប. ។ ការបង្ហាញឱ្យឃើញតួនាទីការទូតរបស់ប្រទេស ។
ភ្នំពេញ ថ្ងៃទី ២១ ខែ មិថុនា ឆ្នាំ ២០២៥ ។
ជូនចំពោះ ឯកឧត្ដម ។ ឧបនាយករដ្ឋមន្ត្រី ។ ជំទាវ ។
ចំណុចសំខាន់ស្ដីពីការសម្ដែងគម្រោង ។ ការបញ្ជាក់ ។
"""

# ============================================================================
# SECTION 1 – Fixed Khmer document strings
#   Used for header / footer / title sections that must look official.
# ============================================================================

MINISTRY_NAMES = [
    ["ព្រះរាជាណាចក្រកម្ពុជា", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងសេដ្ឋកិច្ចនិងហិរញ្ញវត្ថុ", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងអប់រំ យុវជន និងកីឡា", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងសុខាភិបាល", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងការងារ និងបណ្ដុះបណ្ដាលវិជ្ជាជីវៈ", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងព័ត៌មាន", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងទេសចរណ៍", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងរៀបចំដែនដីនគរូបនីយកម្ម", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងពាណិជ្ជកម្ម", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
    ["ក្រសួងបរិស្ថាន", "ជាតិ សាសនា ព្រះមហាក្សត្រ"],
]

INSTITUTION_LINES = [
    "លេខ​ ០២/សសស/ហហ",
    "លិខិតលេខ ១២៥ សសស",
    "លិខិតលេខ ០០៣ ក.ស.ហ",
    "គណៈកម្មការ​ ​ចំនួន​ ​១០​ ​រូប",
    "ក្រសួង​-​ទីស្ដីការ",
]

DOCUMENT_TITLES = [
    ["សេចក្ដីប្រកាសព័ត៌មាន", "ស្ដីពីការអភិវឌ្ឍន៍ហេដ្ឋារចនាសម្ព័ន្ធ"],
    ["សេចក្ដីជូនដំណឹង", "ស្ដីពីកិច្ចប្រជុំគណៈរដ្ឋមន្ត្រី"],
    ["សេចក្ដីព្រាងច្បាប់", "ស្ដីពីការគ្រប់គ្រងធនធានធម្មជាតិ"],
    ["សេចក្ដីសម្រេច", "ស្ដីពីការកំណត់ប្រាក់ឈ្នួលអប្បបរមា"],
    ["សាររបស់រដ្ឋមន្ត្រី", "ស្ដីពីការពង្រឹងសន្ដិភាពសង្គម"],
    ["សេចក្ដីប្រកាស", "ស្ដីពីការធ្វើកំណែទម្រង់ការអប់រំ"],
    ["ប្រកាសស្ដីពីការដំឡើងប្រាក់ខែ", "ចាប់ពីខែ​មករា​ ​ឆ្នាំ​ ​២០២៥"],
    ["សេចក្ដីជូនដំណឹងសាធារណៈ", "ស្ដីពីការចុះឈ្មោះទទួលការបោះឆ្នោត"],
]

FOOTER_SIGNATURES = [
    ["ភ្នំពេញ ថ្ងៃទី ០៥ ខែ មករា ឆ្នាំ ២០២៥",  "រដ្ឋមន្ត្រី",         "ឯកឧត្ដម ហ៊ុន ម៉ាណែត"],
    ["ភ្នំពេញ ថ្ងៃទី ១២ ខែ កុម្ភៈ ឆ្នាំ ២០២៥", "ឧបនាយករដ្ឋមន្ត្រី",  "ឯកឧត្ដម ប្រាក់ សុខុន"],
    ["ភ្នំពេញ ថ្ងៃទី ២០ ខែ មីនា ឆ្នាំ ២០២៥",  "រដ្ឋលេខាធិការ",      "ឯកឧត្ដម ចាន់ ដារ៉ា"],
    ["ភ្នំពេញ ថ្ងៃទី ០៨ ខែ មេសា ឆ្នាំ ២០២៥",  "អគ្គនាយក",           "ឯកឧត្ដម នូ ចន្ទ"],
    ["ភ្នំពេញ ថ្ងៃទី ១៥ ខែ ឧសភា ឆ្នាំ ២០២៥",  "នាយករដ្ឋមន្ត្រី",    "ឯកឧត្ដម ហ៊ុន ម៉ាណែត"],
]


# ============================================================================
# SECTION 2 – Colour palettes
# ============================================================================

# Paper-like background colours (RGB)
PAPER_COLORS = [
    (255, 255, 255),   # pure white
    (252, 251, 248),   # warm white
    (250, 249, 245),   # off-white
    (248, 246, 240),   # cream
    (245, 244, 238),   # ivory
    (252, 252, 248),   # cool white
    (250, 248, 242),   # parchment
    (246, 245, 242),   # light grey-white
    (253, 252, 244),   # pale yellow
    (251, 250, 246),   # linen
]

# Ink colours – dark tones only (no colour printing on official docs)
INK_COLORS = [
    (15,  15,  20),    # near-black
    (20,  20,  25),    # very dark grey
    (30,  30,  35),    # dark charcoal
    (10,  10,  45),    # very dark navy
    (25,  10,  10),    # very dark brown-black
]


# ============================================================================
# SECTION 3 – Asset loaders
# ============================================================================

# ---------------------------------------------------------------------------
# Khmer font auto-download
# ---------------------------------------------------------------------------
# URLs for free Noto Sans Khmer fonts (Google Fonts / GitHub releases).
# The list is tried in order; the first successful download is kept.
# All are SIL Open Font License – safe for any use.
# ---------------------------------------------------------------------------
_KHMER_FONT_URLS: list[tuple[str, str]] = [
    (
        "NotoSansKhmer-Regular.ttf",
        "https://github.com/google/fonts/raw/main/ofl/notosanskhmervf/"
        "NotoSansKhmer%5Bwdth%2Cwght%5D.ttf",
    ),
    (
        "NotoSansKhmer-Regular.ttf",
        "https://github.com/notofonts/khmer/releases/download/NotoSansKhmer-v2.005/"
        "NotoSansKhmer.zip",
    ),
]


def _try_download_khmer_font(fonts_dir: Path) -> Optional[Path]:
    """
    Attempt to download a free Khmer font into *fonts_dir*.

    Tries each URL in _KHMER_FONT_URLS in order.  Returns the saved Path on
    success, or None if every attempt fails (network unavailable, etc.).

    The download is intentionally *silent on failure* – the caller decides
    whether to abort or warn.
    """
    import urllib.request
    import urllib.error
    import zipfile
    import io as _io

    fonts_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in _KHMER_FONT_URLS:
        dest = fonts_dir / filename
        if dest.exists():
            return dest   # already present from a previous run
        try:
            print(f"[INFO] Downloading Khmer font from:\n       {url}")
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = resp.read()

            # If the URL points to a zip, extract the first .ttf inside it
            if url.endswith(".zip"):
                with zipfile.ZipFile(_io.BytesIO(data)) as zf:
                    ttf_names = [n for n in zf.namelist()
                                 if n.lower().endswith(".ttf") and "/" not in n]
                    if not ttf_names:
                        # Try any ttf at any depth
                        ttf_names = [n for n in zf.namelist()
                                     if n.lower().endswith(".ttf")]
                    if not ttf_names:
                        continue
                    data = zf.read(ttf_names[0])

            dest.write_bytes(data)
            print(f"[INFO] Font saved → {dest}")
            return dest

        except Exception as exc:
            print(f"[WARNING] Download failed ({url!r}): {exc}")
            continue

    return None


def ensure_khmer_font(fonts_dir: Path) -> None:
    """
    If *fonts_dir* contains no .ttf/.otf files, attempt to download
    Noto Sans Khmer automatically.  Prints a clear error and exits if
    the download also fails – running without a Khmer font produces
    unreadable tofu boxes and worthless training data.
    """
    existing = [p for p in fonts_dir.rglob("*")
                if p.suffix.lower() in {".ttf", ".otf"}]
    if existing:
        return   # fonts already present, nothing to do

    print(
        "\n[WARNING] No Khmer fonts found in '{dir}'.\n"
        "          Pillow's built-in font covers only ASCII – Khmer text\n"
        "          would render as empty boxes (tofu □), making training\n"
        "          data useless.  Attempting automatic download …\n"
        .format(dir=fonts_dir)
    )

    saved = _try_download_khmer_font(fonts_dir)

    if saved is None:
        print(
            "\n[ERROR] Could not download a Khmer font automatically.\n"
            "\n"
            "  Please download one manually and place it in: {dir}\n"
            "\n"
            "  Free options (SIL Open Font License):\n"
            "    • Noto Sans Khmer  – https://fonts.google.com/noto/specimen/Noto+Sans+Khmer\n"
            "    • Hanuman           – https://fonts.google.com/specimen/Hanuman\n"
            "    • Khmer OS          – https://sourceforge.net/projects/khmer/files/Fonts/\n"
            "\n"
            "  Then re-run this script.\n"
            .format(dir=fonts_dir)
        )
        import sys
        sys.exit(1)

    print(f"[✓] Khmer font downloaded successfully → {saved}\n")


def load_fonts(font_dir: str = "./fonts") -> list[Path]:
    """
    Scan *font_dir* recursively for .ttf / .otf files.

    If the directory is empty, calls ensure_khmer_font() which either
    downloads Noto Sans Khmer automatically or exits with a clear error
    message – preventing the silent tofu-box fallback.
    """
    root = Path(font_dir)
    root.mkdir(parents=True, exist_ok=True)   # create dir if missing

    # Auto-download if needed (exits on failure)
    ensure_khmer_font(root)

    all_files = [p for p in root.rglob("*") if p.suffix.lower() in {".ttf", ".otf"}]

    # Keep only fonts that actually contain Khmer glyphs.
    # This prevents silently using a Latin-only font that renders every
    # Khmer codepoint as a tofu box.
    verified: list[Path] = []
    rejected: list[Path] = []
    for fp in all_files:
        if verify_font_has_khmer(fp):
            verified.append(fp)
        else:
            rejected.append(fp)

    if rejected:
        print(f"[WARNING] {len(rejected)} font(s) rejected (no Khmer glyphs): "
              + ", ".join(p.name for p in rejected))
    if not verified and all_files:
        # All fonts were rejected – fall back to using them anyway with a warning
        # (better than crashing with a confusing error)
        print("[WARNING] No font passed Khmer verification; using all fonts anyway. "
              "Expect possible tofu rendering on complex clusters.")
        verified = all_files

    print(f"[i] Fonts: {len(verified)} Khmer-verified  ({font_dir})")
    return verified



def clean_corpus(lines: list[str]) -> list[str]:
    """
    Filter and normalise raw corpus lines so only renderable Khmer text
    survives into the generator.

    The Tesseract corpus is noisy OCR output: many lines are a soup of
    Khmer codepoints, ASCII punctuation, numbers, and broken clusters.
    We apply three tiers of filtering:

    Tier 1 – Hard reject (line is discarded entirely):
      • Fewer than 10 characters after stripping
      • Khmer codepoints make up less than 50% of non-space characters
        (mostly ASCII / noise lines)

    Tier 2 – Soft clean (line is kept but sanitised):
      • Strip leading/trailing whitespace
      • Collapse multiple spaces into one
      • Remove ASCII control characters (< U+0020) and the Unicode
        replacement character U+FFFD
      • Remove lone ASCII characters that are surrounded by Khmer — these
        are typically OCR artefacts (e.g. stray "|", "$", "›", "_")

    Tier 3 – Split long lines:
      • Lines longer than 200 characters are split on Khmer sentence-final
        markers (។ ៕ ៖) or whitespace runs, keeping only segments that
        pass Tier 1 themselves.  This prevents single very-long noisy lines
        from dominating the word-wrap budget.

    Returns a de-duplicated list; preserves relative order.
    """
    import unicodedata, re

    # Khmer Unicode block: U+1780–U+17FF (base chars + diacritics)
    # Also accept U+19E0–U+19FF (Khmer Symbols) and U+A9E0–U+A9FF (Khmer Ext-B)
    def is_khmer(ch: str) -> bool:
        cp = ord(ch)
        return (0x1780 <= cp <= 0x17FF) or (0x19E0 <= cp <= 0x19FF) or (0xA9E0 <= cp <= 0xA9FF)

    # Regex: one or more ASCII "junk" chars surrounded by Khmer on both sides
    _JUNK_ASCII = re.compile(r"(?<=[ក-៿])[!-~_]{1,3}(?=[ក-៿])")
    # Collapse whitespace
    _MULTI_SPACE = re.compile(r"  +")
    # Control characters (except normal space / newline)
    _CONTROL = re.compile("[\x00-\x1F\x7F\ufffd]")
    # Sentence-break points for splitting long lines
    _SPLIT_SEP = re.compile(r"[។៕]\s*|\s{2,}")

    def tier1_ok(text: str) -> bool:
        if len(text) < 10:
            return False
        non_space = [c for c in text if c != " "]
        if not non_space:
            return False
        khmer_ratio = sum(1 for c in non_space if is_khmer(c)) / len(non_space)
        return khmer_ratio >= 0.50

    def tier2_clean(text: str) -> str:
        text = _CONTROL.sub("", text)           # strip control chars
        text = _JUNK_ASCII.sub("", text)        # strip junk ASCII between Khmer
        text = _MULTI_SPACE.sub(" ", text)      # collapse spaces
        return text.strip()

    seen:   set[str] = set()
    result: list[str] = []

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        # Tier 3: split very long lines first
        segments = [raw] if len(raw) <= 200 else _SPLIT_SEP.split(raw)

        for seg in segments:
            seg = tier2_clean(seg)
            if not tier1_ok(seg):
                continue
            if seg in seen:
                continue
            seen.add(seg)
            result.append(seg)

    return result


def load_corpus(corpus_path: str = "./texts/khmer_corpus.txt") -> list[str]:
    """
    Load the text corpus as a list of non-empty lines.
    Creates the file with fallback text if it does not exist.
    """
    path = Path(corpus_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"[INFO] Corpus not found – creating fallback corpus at {path}")
        path.write_text(FALLBACK_CORPUS, encoding="utf-8")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        print("[WARNING] Corpus file is empty. Using fallback text.")
        lines = [ln.strip() for ln in FALLBACK_CORPUS.splitlines() if ln.strip()]

    cleaned = clean_corpus(lines)
    print(f"[i] Corpus: {len(lines)} raw lines → {len(cleaned)} usable after cleaning")
    return cleaned


def pick_font(
    font_files: list[Path],
    size: int,
) -> ImageFont.FreeTypeFont:
    """
    Load a random TrueType font from *font_files* at *size* pixels.

    Raises RuntimeError if font_files is empty – callers must ensure
    ensure_khmer_font() has been called first so we never silently fall
    back to Pillow's ASCII-only bitmap font (which renders all Khmer
    codepoints as tofu boxes □).

    If a specific file cannot be loaded (corrupt / wrong format), it is
    skipped and the next random candidate is tried; RuntimeError is raised
    only when every candidate has been exhausted.
    """
    if not font_files:
        raise RuntimeError(
            "No Khmer font files available.  "
            "Place at least one .ttf/.otf in ./fonts/ and re-run."
        )

    # Shuffle a copy so we try each file at most once before giving up
    candidates = font_files.copy()
    random.shuffle(candidates)
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError as exc:
            print(f"[WARNING] Could not load font '{path}': {exc}  – trying next …")

    raise RuntimeError(
        f"All {len(font_files)} font file(s) failed to load.  "
        "Please check the files in ./fonts/."
    )

def verify_font_has_khmer(font_path: Path) -> bool:
    """
    Return True if the font file at *font_path* actually contains glyphs
    for basic Khmer codepoints.

    We test three representative codepoints:
      U+1780 ក  (base consonant — must be present in any Khmer font)
      U+17D2 ្  (COENG — subscript marker, present in all complete fonts)
      U+17B6 ា  (vowel sign — present in all complete fonts)

    Uses fonttools if available (accurate) or falls back to a Pillow
    render-width heuristic (portable but slightly less reliable).
    """
    test_chars = ["ក", "្", "ា"]

    # ── Method A: fonttools cmap lookup (most reliable) ─────────────────────
    try:
        from fontTools.ttLib import TTFont   # type: ignore
        tt = TTFont(str(font_path), lazy=True)
        cmap = tt.getBestCmap()
        if cmap:
            return all(ord(ch) in cmap for ch in test_chars)
    except Exception:
        pass   # fonttools not installed or font is unreadable → try Method B

    # ── Method B: Pillow render-width heuristic ──────────────────────────────
    # If FreeType has no glyph it substitutes the .notdef glyph (a box).
    # The .notdef glyph always has the same width for every "missing" char,
    # so if all three test chars report the same width they are all probably
    # .notdef boxes.  A real Khmer font gives different widths for ក, ្, ា.
    try:
        fnt = ImageFont.truetype(str(font_path), size=24)
        dummy_img  = Image.new("RGB", (100, 50))
        dummy_draw = ImageDraw.Draw(dummy_img)
        widths = {
            ch: dummy_draw.textlength(ch, font=fnt)
            for ch in test_chars
        }
        # If all three characters have zero or identical widths → tofu font
        unique_widths = set(round(w, 1) for w in widths.values())
        if len(unique_widths) == 1 and list(unique_widths)[0] <= 1.0:
            return False   # all zero → no glyphs at all
        return True        # at least some variation → likely has Khmer
    except Exception:
        return False       # cannot load font at all




# ============================================================================
# SECTION 4 – Text wrapping helper (CRITICAL – Pillow has no auto-wrap)
# ============================================================================

def wrap_text(
    text:      str,
    font:      ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    draw:      ImageDraw.ImageDraw,
) -> list[str]:
    """
    Split *text* into a list of lines where each line fits within *max_width*
    pixels when rendered with *font*.

    Strategy
    --------
    1. Split on whitespace to get individual words.
    2. Accumulate words onto the current line until adding the next word would
       exceed max_width.
    3. When a single word is wider than max_width, force it onto its own line
       (no infinite loop).
    4. Use draw.textlength() for width measurement (Pillow ≥ 9.2).
       Falls back to draw.textbbox() for older versions.

    Returns a list of strings (may be empty if *text* is blank).
    """
    if not text.strip():
        return []

    def line_width(s: str) -> float:
        try:
            return draw.textlength(s, font=font)
        except AttributeError:
            # Pillow < 9.2 fallback
            bbox = draw.textbbox((0, 0), s, font=font)
            return bbox[2] - bbox[0]

    words   = text.split()
    lines:  list[str] = []
    current = ""

    for word in words:
        test = (current + " " + word).strip() if current else word
        if line_width(test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            # If the word alone is too wide, keep it anyway (no infinite loop)
            current = word

    if current:
        lines.append(current)

    return lines


# ============================================================================
# SECTION 5 – YOLO bbox converter
# ============================================================================

def to_yolo(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    pad_x: int = 3,
    pad_y: int = 2,
) -> tuple[float, float, float, float]:
    """
    Convert absolute pixel bbox (x1,y1,x2,y2) to normalised YOLO
    (x_center, y_center, width, height) with optional small padding.
    Values are clamped to [0, 1].
    """
    x1 = max(0,     x1 - pad_x)
    y1 = max(0,     y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1)    / img_w
    h  = (y2 - y1)    / img_h

    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


# ============================================================================
# SECTION 6 – Line renderer
#   Draws one line of text, records its bounding box, and advances the cursor.
# ============================================================================

def draw_line(
    draw:        ImageDraw.ImageDraw,
    text:        str,
    x:           int,
    y:           int,
    font:        ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ink:         tuple[int, int, int],
    img_w:       int,
    img_h:       int,
    align:       str = "left",   # "left" | "center" | "right"
    region_w:    Optional[int] = None,  # width of the layout region for centering
) -> tuple[int, tuple[float, float, float, float]]:
    """
    Draw *text* at (x, y) and return:
        (new_y, yolo_bbox)
    where new_y = y + line_height + small leading gap.

    *align* controls horizontal positioning within *region_w*.
    For centered/right text the draw position is recalculated; the YOLO bbox
    is always derived from the actual rendered glyph position.
    """
    if not text.strip():
        # Blank line – just advance by a small gap
        try:
            _, top, _, bottom = draw.textbbox((0, 0), "ក", font=font)
            h = bottom - top
        except Exception:
            h = 20
        return y + h + 4, None

    # ── Compute glyph metrics ────────────────────────────────────────────────
    try:
        bx1, by1, bx2, by2 = draw.textbbox((x, y), text, font=font, anchor="lt")
    except TypeError:
        bx1, by1, bx2, by2 = draw.textbbox((x, y), text, font=font)

    text_w = bx2 - bx1
    text_h = by2 - by1

    # ── Adjust x for alignment ───────────────────────────────────────────────
    rw = region_w if region_w is not None else (img_w - x)
    if align == "center":
        x = x + (rw - text_w) // 2
    elif align == "right":
        x = x + rw - text_w

    # Re-measure after x adjustment
    try:
        bx1, by1, bx2, by2 = draw.textbbox((x, y), text, font=font, anchor="lt")
    except TypeError:
        bx1, by1, bx2, by2 = draw.textbbox((x, y), text, font=font)

    # ── Draw ─────────────────────────────────────────────────────────────────
    # Wrap in a broad except so a bad glyph cluster (e.g. invalid Khmer
    # stack from OCR noise) never crashes the whole document — we just
    # skip this line and return None as the bbox.
    try:
        try:
            draw.text((x, y), text, font=font, fill=ink, anchor="lt")
        except TypeError:
            draw.text((x, y), text, font=font, fill=ink)
    except Exception as exc:
        # Re-measure a blank-line height so cursor still advances correctly
        try:
            _, _t, _, _b = draw.textbbox((0, 0), "ក", font=font)
            _h = max(4, _b - _t)
        except Exception:
            _h = 20
        return y + _h + 4, None   # skip; no annotation

    # ── Build YOLO annotation ─────────────────────────────────────────────────
    yolo_box = to_yolo(bx1, by1, bx2, by2, img_w, img_h)
    line_height = by2 - by1

    # Sanity-check: if the bbox is degenerate (zero or negative size) the
    # glyph likely did not render — skip the annotation.
    if line_height <= 2 or (bx2 - bx1) <= 2:
        leading = max(4, int(abs(line_height) * 0.20)) if line_height else 4
        return y + line_height + leading, None

    # Leading: ~20% of glyph height
    leading = max(4, int(line_height * 0.20))
    new_y   = by2 + leading

    return new_y, yolo_box


# ============================================================================
# SECTION 7 – Page-geometry & layout configuration
#   All per-document variation is captured in a single DocConfig so every
#   choice is explicit, reproducible, and easy to extend.
# ============================================================================

from dataclasses import dataclass, field as dc_field
from enum import Enum


class LayoutTemplate(Enum):
    """
    Structural blueprints for the document.  Each template changes which
    sections are present, their alignment, and column arrangement — so the
    model sees text lines at many different absolute positions.

    STANDARD       Header (2-col) | Title (centred) | Body (1-col) | Footer (right)
    LETTER         Header (1-col left) | Addressee | Body (1-col) | Sig (right)
    ANNOUNCEMENT   Header (centred) | Large title | Short body | No footer
    REPORT         No header title | Section numbers | Dense body | Footer (left)
    SPARSE         Header (right) | Title (left) | Few body lines | No footer
    TWO_COLUMN     Header (2-col) | Title | Body split into 2 narrow columns
    MEMO           Header minimal | TO/FROM block | Short body | Sig (centre)
    PLAIN          No header | No footer | Body only (pure text block)
    """
    STANDARD    = "standard"
    LETTER      = "letter"
    ANNOUNCEMENT = "announcement"
    REPORT      = "report"
    SPARSE      = "sparse"
    TWO_COLUMN  = "two_column"
    MEMO        = "memo"
    PLAIN       = "plain"


# All supported page sizes (width, height) in pixels.
# Covers A4-portrait, A4-landscape, A5, Letter, half-page, tall-narrow, wide-short.
PAGE_SIZES: list[tuple[int, int]] = [
    (800,  1130),   # A4 portrait  (baseline)
    (760,  1075),   # A4 portrait  (slightly smaller)
    (850,  1180),   # A4 portrait  (slightly larger)
    (1130,  800),   # A4 landscape
    (566,   800),   # A5 portrait
    (816,  1056),   # US Letter portrait
    (1056,  816),   # US Letter landscape
    (640,  1000),   # tall narrow  (phone-scan feel)
    (900,   640),   # wide short   (cropped scan)
    (720,   960),   # mid-size portrait
    (600,   850),   # small document
    (950,  1250),   # large A4+
]


@dataclass
class DocConfig:
    """
    One randomly sampled configuration that drives a single document render.
    Everything that varies between documents lives here.
    """
    # ── Canvas ────────────────────────────────────────────────────────────────
    page_w:  int
    page_h:  int
    bg_color: tuple[int, int, int]
    ink:      tuple[int, int, int]

    # ── Margins (pixels) ──────────────────────────────────────────────────────
    ml: int   # left
    mr: int   # right
    mt: int   # top
    mb: int   # bottom

    # ── Layout ────────────────────────────────────────────────────────────────
    template:      LayoutTemplate
    n_paragraphs:  int         # body paragraph count
    indent:        int         # first-line indent (0 = no indent)
    line_spacing:  float       # multiplier on top of natural leading (1.0–2.0)
    para_spacing:  int         # extra pixels between paragraphs

    # ── Column layout ─────────────────────────────────────────────────────────
    n_body_cols:   int         # 1 or 2
    col_gap:       int         # pixels between columns (only for n_body_cols=2)

    # ── Sections present/absent ───────────────────────────────────────────────
    has_header:    bool
    has_title:     bool
    has_footer:    bool
    has_divider:   bool        # thin horizontal rule below header

    # ── Alignment choices ─────────────────────────────────────────────────────
    header_align:  str         # "split" | "left" | "center" | "right"
    title_align:   str         # "center" | "left" | "right"
    footer_align:  str         # "center" | "left" | "right"
    body_align:    str         # "left" | "justify_stub" (left with ragged right)

    # ── Font sizes ────────────────────────────────────────────────────────────
    header_size:   int
    title_size:    int
    body_size:     int
    footer_size:   int

    # ── Second font file (mixed-font documents) ───────────────────────────────
    # None = use the same font for everything
    alt_font_file: Optional[Path]

    @classmethod
    def sample(cls, font_files: list[Path]) -> "DocConfig":
        """
        Randomly draw a DocConfig.  Called once per document.
        """
        pw, ph = random.choice(PAGE_SIZES)

        # Margins: scale proportionally with page size so small pages don't
        # have absurdly wide margins.
        scale = pw / 800
        ml = int(random.randint(45, 90) * scale)
        mr = int(random.randint(45, 90) * scale)
        mt = int(random.randint(45, 85) * scale)
        mb = int(random.randint(40, 80) * scale)

        template = random.choice(list(LayoutTemplate))

        # Font sizes: scaled with page width so small pages get smaller text
        fs_scale = max(0.7, pw / 800)
        h_size = int(random.randint(14, 22) * fs_scale)
        t_size = int(random.randint(18, 28) * fs_scale)
        b_size = int(random.randint(13, 20) * fs_scale)
        f_size = int(random.randint(12, 18) * fs_scale)

        # Two-column body for templates that support it
        two_col = (template == LayoutTemplate.TWO_COLUMN)
        n_cols  = 2 if two_col else 1
        col_gap = random.randint(20, 40) if two_col else 0

        # Occasionally mix two fonts
        alt = None
        if len(font_files) >= 2 and random.random() < 0.35:
            alt = random.choice([f for f in font_files
                                  if f != font_files[0]])

        # Template-specific overrides
        has_hdr = template not in (LayoutTemplate.PLAIN,)
        has_ttl = template not in (LayoutTemplate.PLAIN, LayoutTemplate.REPORT)
        has_ftr = template not in (LayoutTemplate.ANNOUNCEMENT,
                                   LayoutTemplate.SPARSE,
                                   LayoutTemplate.PLAIN)

        # Header alignment driven by template
        hdr_align_map = {
            LayoutTemplate.STANDARD:     "split",
            LayoutTemplate.LETTER:       "left",
            LayoutTemplate.ANNOUNCEMENT: "center",
            LayoutTemplate.REPORT:       "left",
            LayoutTemplate.SPARSE:       "right",
            LayoutTemplate.TWO_COLUMN:   "split",
            LayoutTemplate.MEMO:         "left",
            LayoutTemplate.PLAIN:        "left",
        }
        ttl_align_map = {
            LayoutTemplate.STANDARD:     "center",
            LayoutTemplate.LETTER:       "left",
            LayoutTemplate.ANNOUNCEMENT: "center",
            LayoutTemplate.REPORT:       "left",
            LayoutTemplate.SPARSE:       "left",
            LayoutTemplate.TWO_COLUMN:   "center",
            LayoutTemplate.MEMO:         "center",
            LayoutTemplate.PLAIN:        "left",
        }
        ftr_align_map = {
            LayoutTemplate.STANDARD:     "right",
            LayoutTemplate.LETTER:       "right",
            LayoutTemplate.ANNOUNCEMENT: "center",
            LayoutTemplate.REPORT:       "left",
            LayoutTemplate.SPARSE:       "right",
            LayoutTemplate.TWO_COLUMN:   "right",
            LayoutTemplate.MEMO:         "center",
            LayoutTemplate.PLAIN:        "left",
        }

        return cls(
            page_w        = pw,
            page_h        = ph,
            bg_color      = random.choice(PAPER_COLORS),
            ink           = random.choice(INK_COLORS),
            ml=ml, mr=mr, mt=mt, mb=mb,
            template      = template,
            n_paragraphs  = random.randint(2, 7),
            indent        = random.choice([0, 0, 20, 30, 40]),  # 40% no-indent
            line_spacing  = random.uniform(1.0, 1.8),
            para_spacing  = random.randint(6, 22),
            n_body_cols   = n_cols,
            col_gap       = col_gap,
            has_header    = has_hdr,
            has_title     = has_ttl,
            has_footer    = has_ftr,
            has_divider   = random.random() < 0.65,
            header_align  = hdr_align_map[template],
            title_align   = ttl_align_map[template],
            footer_align  = ftr_align_map[template],
            body_align    = "left",
            header_size   = max(10, h_size),
            title_size    = max(12, t_size),
            body_size     = max(10, b_size),
            footer_size   = max(10, f_size),
            alt_font_file = alt,
        )


# ============================================================================
# SECTION 7b – Augmentation pipeline
#   Two pipelines: standard and "heavy" (simulates poor scans / old paper).
#   A document randomly gets one of the two.
# ============================================================================

def _build_standard_aug() -> A.Compose:
    """Moderate photometric augmentation for clean-scan simulation."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.70),
        A.GaussianBlur(blur_limit=(3, 3), p=0.25),
        A.GaussNoise(std_range=(0.004, 0.020), p=0.40),
        A.ImageCompression(quality_range=(72, 97), p=0.35),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 1),
            shadow_dimension=4, shadow_intensity_range=(0.02, 0.07), p=0.25,
        ),
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=8, val_shift_limit=12, p=0.30),
        A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.20),
    ])


def _build_heavy_aug() -> A.Compose:
    """Heavy photometric augmentation simulating old/worn/poor-scan documents."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.85),
        A.GaussianBlur(blur_limit=(3, 5), p=0.50),
        A.GaussNoise(std_range=(0.015, 0.055), p=0.60),
        A.ImageCompression(quality_range=(45, 75), p=0.55),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2),
            shadow_dimension=5, shadow_intensity_range=(0.05, 0.18), p=0.50,
        ),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=25, p=0.60),
        A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18, p=0.40),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.30),
        A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.25),
    ])


def build_augmentation_pipeline() -> tuple[A.Compose, A.Compose]:
    """
    Return (standard_pipeline, heavy_pipeline).
    generate_document() randomly picks one per image (80/20 split).
    """
    return _build_standard_aug(), _build_heavy_aug()


def augment_image(
    img_pil:  Image.Image,
    standard: A.Compose,
    heavy:    A.Compose,
) -> Image.Image:
    """
    Apply a randomly chosen augmentation pipeline.
    Also applies two Pillow-level post-processes:
      • Paper texture overlay (faint grain on the background)
      • Optional edge-darkening (simulates scanner vignette)
    """
    # ── Pillow pre-process: paper grain ──────────────────────────────────────
    img_np = np.array(img_pil)

    # Faint uniform noise to simulate paper fibre texture (always applied)
    grain_strength = random.uniform(0, 6)
    if grain_strength > 1:
        grain = np.random.normal(0, grain_strength, img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # ── Optional edge vignette (scanner lamp falloff) ─────────────────────────
    if random.random() < 0.30:
        h, w = img_np.shape[:2]
        # Build a radial brightness mask: bright centre, slightly darker edges
        cx, cy = w / 2, h / 2
        xs = np.linspace(0, w, w)
        ys = np.linspace(0, h, h)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        strength = random.uniform(0.04, 0.12)
        mask = (1.0 - dist * strength).clip(0.85, 1.0).astype(np.float32)
        img_np = np.clip(
            (img_np.astype(np.float32) * mask[:, :, None]), 0, 255
        ).astype(np.uint8)

    # ── Optional fold/crease line ─────────────────────────────────────────────
    if random.random() < 0.18:
        h, w = img_np.shape[:2]
        if random.random() < 0.5:
            # Horizontal crease
            y_crease = random.randint(h // 4, 3 * h // 4)
            thickness = random.randint(1, 3)
            alpha = random.uniform(0.04, 0.10)
            img_np[y_crease:y_crease+thickness, :] = np.clip(
                img_np[y_crease:y_crease+thickness, :].astype(np.float32) * (1 - alpha), 0, 255
            ).astype(np.uint8)
        else:
            # Vertical crease
            x_crease = random.randint(w // 4, 3 * w // 4)
            thickness = random.randint(1, 2)
            alpha = random.uniform(0.03, 0.08)
            img_np[:, x_crease:x_crease+thickness] = np.clip(
                img_np[:, x_crease:x_crease+thickness].astype(np.float32) * (1 - alpha), 0, 255
            ).astype(np.uint8)

    # ── Albumentations pipeline ───────────────────────────────────────────────
    pipeline = heavy if random.random() < 0.20 else standard
    out_np   = pipeline(image=img_np)["image"]
    return Image.fromarray(out_np)


# ============================================================================
# SECTION 8b – Object overlay (stamps, logos, non-text elements)
#
# These assets are pasted onto the document WITHOUT being annotated, so the
# model learns that stamps / logos / decorative elements are NOT text lines.
#
# Key rules
# ----------
# • Objects are composited BEFORE albumentations so they receive the same
#   scan-simulation artefacts (blur, noise, JPEG compression) as the text.
# • Only the object pixel itself is modified; yolo_boxes is never touched.
# • PNG files with an alpha channel are blended using their alpha mask.
#   Opaque images (JPEG, etc.) are blended with a random opacity.
# • The object can be rotated freely (it is not annotated, so bbox validity
#   is irrelevant for the pasted object itself).
# • Stamp-mode: a semi-transparent coloured tint is applied to simulate an
#   ink stamp overtop of text.
# ============================================================================

def load_object_pool(objects_dir: str = "./objects") -> dict[str, list[Path]]:
    """
    Scan *objects_dir* for image files and return a categorised pool.

    Subdirectory names drive placement — the script looks for these names
    anywhere in the relative path (case-insensitive):

        stamps/   → ink stamps: bottom-right corner, slight rotation, tinted
        logos/    → logos / emblems: top-centre header area, no rotation
        seals/    → official seals: top-centre or beside header text
        misc/     → anything else: treated as a stamp

    Files not inside a recognised subfolder go into "misc".

    Returns
    -------
    dict with keys "stamps", "logos", "seals", "misc",
    each mapping to a (possibly empty) list of Paths.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(objects_dir)
    pool: dict[str, list[Path]] = {
        "stamps": [], "logos": [], "seals": [], "misc": [],
    }
    if not root.is_dir():
        print(f"[INFO] Objects directory '{objects_dir}' not found — "
              "no stamp/logo overlays will be added.")
        return pool

    for p in root.rglob("*"):
        if p.suffix.lower() not in supported or not p.is_file():
            continue
        # Check relative path for category keyword
        rel_lower = str(p.relative_to(root)).lower()
        if "stamp" in rel_lower:
            pool["stamps"].append(p)
        elif "logo" in rel_lower:
            pool["logos"].append(p)
        elif "seal" in rel_lower:
            pool["seals"].append(p)
        else:
            pool["misc"].append(p)

    total = sum(len(v) for v in pool.values())
    if total == 0:
        print(f"[INFO] Objects directory '{objects_dir}' exists but is empty.")
    else:
        print(f"[i] Object assets: {total} total  "
              f"(stamps={len(pool['stamps'])}  logos={len(pool['logos'])}  "
              f"seals={len(pool['seals'])}  misc={len(pool['misc'])})")
    return pool


def _load_object_as_rgba(
    obj_path: Path,
    target_long_side: int,
) -> "Image.Image | None":
    """
    Load *obj_path* as RGBA and resize so the longer side == *target_long_side*.
    Returns None on any error.
    """
    try:
        img = Image.open(obj_path).convert("RGBA")
    except Exception:
        return None
    w, h   = img.size
    scale  = target_long_side / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def _apply_ink_tint(
    obj_rgba: "Image.Image",
    tint: tuple[int, int, int],
    strength: float = 0.65,
) -> "Image.Image":
    """
    Recolour *obj_rgba* toward *tint* while preserving its alpha channel.
    *strength* 0–1 controls how dominant the tint colour is.
    """
    ow, oh = obj_rgba.size
    r, g, b, a = obj_rgba.split()
    r = Image.blend(r.convert("RGB").split()[0],
                    Image.new("L", (ow, oh), tint[0]), strength)
    g = Image.blend(g.convert("RGB").split()[0],
                    Image.new("L", (ow, oh), tint[1]), strength)
    b = Image.blend(b.convert("RGB").split()[0],
                    Image.new("L", (ow, oh), tint[2]), strength)
    return Image.merge("RGBA", (r, g, b, a))


def _set_opacity(obj_rgba: "Image.Image", opacity: float) -> "Image.Image":
    """Scale the alpha channel of *obj_rgba* by *opacity* (0–1)."""
    r, g, b, a = obj_rgba.split()
    a = a.point(lambda p: int(p * opacity))
    return Image.merge("RGBA", (r, g, b, a))


def _has_transparency(obj_rgba: "Image.Image") -> bool:
    """Return True if the image already has meaningful alpha variation."""
    a_data = obj_rgba.getchannel("A").tobytes()
    # Sample 400 bytes spread across the alpha channel
    step    = max(1, len(a_data) // 400)
    sampled = a_data[::step]
    return any(b < 240 for b in sampled)


# ---------------------------------------------------------------------------
# Placement zone definitions
#
# Each zone is a callable: (page_w, page_h, obj_w, obj_h, margin) -> (px, py)
# ---------------------------------------------------------------------------
def _zone_top_centre(pw, ph, ow, oh, mg):
    """Centre horizontally in the top header band."""
    px = (pw - ow) // 2 + random.randint(-int(pw * 0.05), int(pw * 0.05))
    py = random.randint(mg, mg + int(ph * 0.10))
    return px, py

def _zone_top_left(pw, ph, ow, oh, mg):
    """Top-left corner — typical for ministry logos on letterhead."""
    px = random.randint(mg, mg + int(pw * 0.08))
    py = random.randint(mg, mg + int(ph * 0.08))
    return px, py

def _zone_top_right(pw, ph, ow, oh, mg):
    """Top-right corner — 'Nation Religion King' emblem side."""
    px = pw - ow - random.randint(mg, mg + int(pw * 0.08))
    py = random.randint(mg, mg + int(ph * 0.08))
    return px, py

def _zone_bottom_right(pw, ph, ow, oh, mg):
    """Bottom-right — most common approval / received stamp position."""
    px = pw - ow - random.randint(mg, mg + int(pw * 0.10))
    py = ph - oh - random.randint(mg, mg + int(ph * 0.10))
    return px, py

def _zone_bottom_left(pw, ph, ow, oh, mg):
    """Bottom-left — 'received' or date stamp."""
    px = random.randint(mg, mg + int(pw * 0.12))
    py = ph - oh - random.randint(mg, mg + int(ph * 0.10))
    return px, py

def _zone_bottom_centre(pw, ph, ow, oh, mg):
    """Bottom centre — sometimes used for official seals under a signature."""
    px = (pw - ow) // 2 + random.randint(-int(pw * 0.06), int(pw * 0.06))
    py = ph - oh - random.randint(mg, mg + int(ph * 0.12))
    return px, py

def _zone_middle_margin_right(pw, ph, ow, oh, mg):
    """Right margin mid-page — side seal or annotation stamp."""
    px = pw - ow - random.randint(mg, mg + int(pw * 0.05))
    py = int(ph * 0.35) + random.randint(0, int(ph * 0.25))
    return px, py


# ---------------------------------------------------------------------------
# Per-category placement blueprints
# Each blueprint is a list of (zone_fn, probability_weight) tuples.
# ---------------------------------------------------------------------------
_LOGO_ZONES = [
    (_zone_top_centre,         50),   # centred between headers
    (_zone_top_left,           30),   # left side of letterhead
    (_zone_top_right,          20),   # right side of letterhead
]

_SEAL_ZONES = [
    (_zone_top_centre,         40),   # national emblem between header columns
    (_zone_bottom_centre,      35),   # below signature block
    (_zone_bottom_right,       15),
    (_zone_top_left,           10),
]

_STAMP_ZONES = [
    (_zone_bottom_right,       45),   # approval stamp
    (_zone_bottom_left,        25),   # received stamp
    (_zone_middle_margin_right, 20),  # side annotation
    (_zone_bottom_centre,      10),
]

_MISC_ZONES  = _STAMP_ZONES   # treat unknown objects as stamps


# ---------------------------------------------------------------------------
# Ink tint palettes per category
# ---------------------------------------------------------------------------
_STAMP_TINTS = [
    (180,  25,  25),   # red — most common official stamp colour
    ( 25,  50, 180),   # blue — also very common
    (100,  25, 150),   # purple
    ( 25, 120,  55),   # green (less common)
]
_SEAL_TINTS  = [
    ( 25,  50, 180),   # blue (national seal is usually blue/gold)
    (160, 120,  20),   # gold/ochre
    (180,  25,  25),   # red
]
_LOGO_TINTS  = []   # logos keep their original colours — do NOT tint them


def _pick_zone(blueprints):
    """Weighted random zone picker."""
    fns, weights = zip(*blueprints)
    return random.choices(fns, weights=weights, k=1)[0]


def _boxes_overlap(
    ax1: int, ay1: int, ax2: int, ay2: int,
    bx1: int, by1: int, bx2: int, by2: int,
    gap: int = 6,
) -> bool:
    """
    Return True if rectangle A (expanded by *gap* px on all sides) intersects B.
    The gap creates a small clear zone so objects never visually touch text.
    """
    return not (
        ax2 + gap <= bx1 or bx2 + gap <= ax1 or
        ay2 + gap <= by1 or by2 + gap <= ay1
    )


def _yolo_to_pixels(
    cx: float, cy: float, w: float, h: float,
    page_w: int, page_h: int,
) -> tuple[int, int, int, int]:
    """Convert normalised YOLO (cx,cy,w,h) to absolute pixel (x1,y1,x2,y2)."""
    x1 = int((cx - w / 2) * page_w)
    y1 = int((cy - h / 2) * page_h)
    x2 = int((cx + w / 2) * page_w)
    y2 = int((cy + h / 2) * page_h)
    return x1, y1, x2, y2


def paste_objects_on_canvas(
    canvas:       "Image.Image",
    object_files: "dict[str, list[Path]] | list[Path]",
    page_w:       int,
    page_h:       int,
    text_boxes:   "list[tuple[float,float,float,float]] | None" = None,
) -> "Image.Image":
    """
    Overlay stamps, logos, and seals onto *canvas* — NEVER overlapping text.

    Overlap guarantee
    -----------------
    Every candidate position is validated against:
      1. All rendered text line boxes (*text_boxes*, YOLO-normalised)
      2. All objects already placed in this document
    If no clear position is found within MAX_TRIES attempts the object is
    silently skipped — it is always better to omit an object than to cover text.

    Rendering — no augmentation on objects
    ----------------------------------------
    No rotation, no tint, no opacity change.  Objects are pasted at full
    fidelity using their own alpha channel (PNG) or with white-to-transparent
    conversion for opaque images (JPEG/BMP scanned on a white background).

    Placement zones
    ---------------
    logos  top-centre 50%, top-left 30%, top-right 20%
    seals  top-centre 40%, below-signature 35%, bottom-right 15%, top-left 10%
    stamps bottom-right 45%, bottom-left 25%, right-margin 20%, bottom-centre 10%
    """
    # Normalise pool
    if isinstance(object_files, list):
        pool: dict[str, list[Path]] = {
            "stamps": [], "logos": [], "seals": [], "misc": object_files,
        }
    else:
        pool = object_files

    total_objects = sum(len(v) for v in pool.values())
    if total_objects == 0 or random.random() > 0.55:
        return canvas

    # Convert text boxes to absolute pixels once
    text_px: list[tuple[int, int, int, int]] = [
        _yolo_to_pixels(*b, page_w, page_h) for b in (text_boxes or [])
    ]

    base   = canvas.convert("RGBA")
    short  = min(page_w, page_h)
    margin = max(8, int(short * 0.025))

    candidates: list[tuple[str, Path]] = [
        (cat, p) for cat, files in pool.items() for p in files
    ]
    n_objects = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    chosen    = random.sample(candidates, min(n_objects, len(candidates)))

    placed_px: list[tuple[int, int, int, int]] = []
    MAX_TRIES = 12

    for cat, obj_path in chosen:

        # 1. Size
        if cat == "logos":
            size = int(short * random.uniform(0.15, 0.28))
        elif cat == "seals":
            size = int(short * random.uniform(0.12, 0.22))
        else:
            size = int(short * random.uniform(0.08, 0.18))

        obj = _load_object_as_rgba(obj_path, size)
        if obj is None:
            continue

        ow, oh = obj.size

        # 2. White-to-transparent for opaque images (scanned logo on white BG)
        if not _has_transparency(obj):
            obj_arr = np.array(obj)
            r_ch = obj_arr[:, :, 0]
            g_ch = obj_arr[:, :, 1]
            b_ch = obj_arr[:, :, 2]
            white_mask = (r_ch > 230) & (g_ch > 230) & (b_ch > 230)
            obj_arr[:, :, 3] = np.where(white_mask, 0, 255).astype(np.uint8)
            obj = Image.fromarray(obj_arr.astype(np.uint8), "RGBA")

        # 3. Pick zone and find a clear (non-overlapping) position
        if cat == "logos":
            blueprints = _LOGO_ZONES
        elif cat == "seals":
            blueprints = _SEAL_ZONES
        else:
            blueprints = _MISC_ZONES

        placed = False
        for _attempt in range(MAX_TRIES):
            zone_fn = _pick_zone(blueprints)
            px, py  = zone_fn(page_w, page_h, ow, oh, margin)
            px = max(0, min(px, page_w - ow))
            py = max(0, min(py, page_h - oh))
            ox1, oy1, ox2, oy2 = px, py, px + ow, py + oh

            # Reject if it overlaps any text line
            if any(_boxes_overlap(ox1, oy1, ox2, oy2, *tb) for tb in text_px):
                continue
            # Reject if it overlaps a previously placed object
            if any(_boxes_overlap(ox1, oy1, ox2, oy2, *pb) for pb in placed_px):
                continue

            placed = True
            break

        if not placed:
            continue   # give up on this object — never overlap text

        # 4. Composite cleanly
        base.paste(obj, (px, py), mask=obj.split()[3])
        placed_px.append((ox1, oy1, ox2, oy2))

    return base.convert("RGB")



# ============================================================================
# SECTION 8 – Document compositor
# ============================================================================

def _resolve_font(
    cfg:        DocConfig,
    font_files: list[Path],
    section:    str,          # "header" | "title" | "body" | "footer"
) -> ImageFont.FreeTypeFont:
    """
    Pick the right font file for *section*.
    If cfg.alt_font_file is set, header & title use the primary font and
    body & footer use the alt font (or vice-versa, randomly decided once
    per document in DocConfig.sample).
    """
    size_map = {
        "header": cfg.header_size,
        "title":  cfg.title_size,
        "body":   cfg.body_size,
        "footer": cfg.footer_size,
    }
    size = size_map.get(section, cfg.body_size)

    use_alt = (
        cfg.alt_font_file is not None
        and section in ("body", "footer")
    )
    file_list = [cfg.alt_font_file] if use_alt else font_files
    return pick_font(file_list, size)


def _draw_section_lines(
    draw:       ImageDraw.ImageDraw,
    texts:      list[str],
    x_start:    int,
    y_start:    int,
    font:       ImageFont.FreeTypeFont,
    ink:        tuple[int, int, int],
    max_w:      int,
    page_w:     int,
    page_h:     int,
    align:      str,
    line_spacing: float = 1.0,
) -> tuple[int, list[tuple[float, float, float, float]]]:
    """
    Wrap and draw a list of raw text strings as consecutive lines.
    Returns (new_y, list_of_yolo_boxes).
    """
    boxes: list[tuple[float, float, float, float]] = []
    y = y_start

    for raw in texts:
        wrapped = wrap_text(raw, font, max_w, draw)
        for line in wrapped:
            new_y, bbox = draw_line(
                draw, line, x_start, y, font, ink,
                page_w, page_h, align=align, region_w=max_w,
            )
            if bbox:
                boxes.append(bbox)
            # Apply line-spacing multiplier on top of natural advance
            natural_advance = new_y - y
            y = y + int(natural_advance * line_spacing)

    return y, boxes


def _generate_memo_block(
    draw:     ImageDraw.ImageDraw,
    font:     ImageFont.FreeTypeFont,
    ink:      tuple[int, int, int],
    ml:       int,
    y:        int,
    text_w:   int,
    page_w:   int,
    page_h:   int,
    corpus:   list[str],
    line_spacing: float,
) -> tuple[int, list[tuple[float, float, float, float]]]:
    """
    Draw a TO / FROM / SUBJECT block typical of a Khmer memo.
    Returns (new_y, boxes).
    """
    # Pick short labels; use corpus words as values
    subjects = random.sample(corpus, min(3, len(corpus)))
    memo_lines = [
        f"ជូនចំពោះ ​: {subjects[0][:40] if subjects else 'ឯកឧត្ដម'}",
        f"ពី        : ក្រសួងសេដ្ឋកិច្ច",
        f"ប្រធានបទ : {subjects[1][:50] if len(subjects) > 1 else 'ការអភិវឌ្ឍន៍'}",
    ]
    return _draw_section_lines(
        draw, memo_lines, ml, y, font, ink, text_w,
        page_w, page_h, align="left", line_spacing=line_spacing,
    )


def generate_document(
    font_files:   list[Path],
    corpus_lines: list[str],
    aug_standard: A.Compose,
    aug_heavy:    A.Compose,
    cfg:          Optional[DocConfig] = None,
    object_files: list[Path] = None,
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """
    Generate one synthetic official document image.

    Parameters
    ----------
    font_files   : verified Khmer TTF/OTF files
    corpus_lines : cleaned Khmer sentences
    aug_standard : standard albumentations pipeline
    aug_heavy    : heavy (degraded scan) pipeline
    cfg          : pre-built DocConfig; if None, one is sampled randomly

    Returns
    -------
    (PIL RGB Image, list of YOLO (cx, cy, w, h) tuples), class_id always 0.
    """
    if cfg is None:
        cfg = DocConfig.sample(font_files)

    # ── Canvas ────────────────────────────────────────────────────────────────
    canvas = Image.new("RGB", (cfg.page_w, cfg.page_h), cfg.bg_color)
    draw   = ImageDraw.Draw(canvas)

    # ── Derived geometry ──────────────────────────────────────────────────────
    text_w    = cfg.page_w - cfg.ml - cfg.mr
    footer_h  = cfg.footer_size * 4 + 20   # enough for 3 footer lines
    footer_y0 = cfg.page_h - cfg.mb - footer_h

    yolo_boxes: list[tuple[float, float, float, float]] = []
    y = cfg.mt

    # =====================================================================
    # BLOCK A – Header
    # =====================================================================
    if cfg.has_header:
        hdr_font = _resolve_font(cfg, font_files, "header")
        ministry = random.choice(MINISTRY_NAMES)
        min_name = ministry[0]
        nrk_line = ministry[1]
        inst_ref = random.choice(INSTITUTION_LINES) if random.random() < 0.6 else None

        if cfg.header_align == "split":
            # ── Two-column header ──────────────────────────────────────────
            left_lines  = [min_name] + ([inst_ref] if inst_ref else [])
            right_lines = [nrk_line]
            right_x = cfg.page_w // 2
            right_w = cfg.page_w - right_x - cfg.mr

            y_left = y
            y_left, boxes = _draw_section_lines(
                draw, left_lines, cfg.ml, y_left, hdr_font, cfg.ink,
                text_w // 2 - 10, cfg.page_w, cfg.page_h,
                align="left", line_spacing=cfg.line_spacing,
            )
            yolo_boxes.extend(boxes)

            y_right = y
            y_right, boxes = _draw_section_lines(
                draw, right_lines, right_x, y_right, hdr_font, cfg.ink,
                right_w, cfg.page_w, cfg.page_h,
                align="center", line_spacing=cfg.line_spacing,
            )
            yolo_boxes.extend(boxes)
            y = max(y_left, y_right)

        elif cfg.header_align == "center":
            # ── Centred header (announcements) ─────────────────────────────
            header_lines = [min_name, nrk_line] + ([inst_ref] if inst_ref else [])
            y, boxes = _draw_section_lines(
                draw, header_lines, cfg.ml, y, hdr_font, cfg.ink,
                text_w, cfg.page_w, cfg.page_h,
                align="center", line_spacing=cfg.line_spacing,
            )
            yolo_boxes.extend(boxes)

        elif cfg.header_align == "right":
            # ── Right-aligned header (sparse/formal) ───────────────────────
            header_lines = [nrk_line, min_name] + ([inst_ref] if inst_ref else [])
            y, boxes = _draw_section_lines(
                draw, header_lines, cfg.ml, y, hdr_font, cfg.ink,
                text_w, cfg.page_w, cfg.page_h,
                align="right", line_spacing=cfg.line_spacing,
            )
            yolo_boxes.extend(boxes)

        else:   # "left"
            header_lines = [min_name] + ([inst_ref] if inst_ref else [])
            y, boxes = _draw_section_lines(
                draw, header_lines, cfg.ml, y, hdr_font, cfg.ink,
                text_w, cfg.page_w, cfg.page_h,
                align="left", line_spacing=cfg.line_spacing,
            )
            yolo_boxes.extend(boxes)

        y += random.randint(8, 20)

        # Optional divider
        if cfg.has_divider:
            draw.line(
                [(cfg.ml, y), (cfg.page_w - cfg.mr, y)],
                fill=cfg.ink, width=1,
            )
            y += random.randint(8, 16)

    # =====================================================================
    # BLOCK B – Memo TO/FROM block (only for MEMO template)
    # =====================================================================
    if cfg.template == LayoutTemplate.MEMO:
        memo_font = _resolve_font(cfg, font_files, "body")
        y, boxes = _generate_memo_block(
            draw, memo_font, cfg.ink,
            cfg.ml, y, text_w, cfg.page_w, cfg.page_h,
            corpus_lines, cfg.line_spacing,
        )
        yolo_boxes.extend(boxes)
        y += random.randint(10, 20)

    # =====================================================================
    # BLOCK C – Title
    # =====================================================================
    if cfg.has_title:
        ttl_font  = _resolve_font(cfg, font_files, "title")
        ttl_entry = random.choice(DOCUMENT_TITLES)
        y, boxes  = _draw_section_lines(
            draw, ttl_entry, cfg.ml, y, ttl_font, cfg.ink,
            text_w, cfg.page_w, cfg.page_h,
            align=cfg.title_align, line_spacing=cfg.line_spacing,
        )
        yolo_boxes.extend(boxes)
        y += random.randint(12, 26)

    # =====================================================================
    # BLOCK D – Body paragraphs
    # =====================================================================
    body_font = _resolve_font(cfg, font_files, "body")

    # Sample sentences; weight toward using more of the corpus for variety
    pool_size    = min(len(corpus_lines), random.randint(20, len(corpus_lines)))
    all_sents    = random.sample(corpus_lines, pool_size)
    sents_per_p  = max(1, pool_size // max(1, cfg.n_paragraphs))

    if cfg.n_body_cols == 2:
        # ── Two-column body ────────────────────────────────────────────────
        col_w  = (text_w - cfg.col_gap) // 2
        col1_x = cfg.ml
        col2_x = cfg.ml + col_w + cfg.col_gap

        # Split sentences evenly across the two columns
        half = len(all_sents) // 2
        col1_sents = all_sents[:half]
        col2_sents = all_sents[half:]

        def render_col(sents, col_x, col_width):
            cy_ = y
            boxes_ = []
            n_p = max(1, cfg.n_paragraphs // 2)
            s_per_p = max(1, len(sents) // n_p)
            for p in range(n_p):
                if cy_ + cfg.body_size >= footer_y0:
                    break
                seg = sents[p * s_per_p: (p + 1) * s_per_p + random.randint(0, 1)]
                para = " ".join(seg)
                if not para.strip():
                    continue
                wrapped = wrap_text(para, body_font, col_width, draw)
                for li, wl in enumerate(wrapped):
                    if cy_ + cfg.body_size >= footer_y0:
                        break
                    x_pos = col_x + (cfg.indent if li == 0 else 0)
                    new_y, bbox = draw_line(
                        draw, wl, x_pos, cy_, body_font, cfg.ink,
                        cfg.page_w, cfg.page_h, align="left",
                    )
                    if bbox:
                        boxes_.append(bbox)
                    cy_ = cy_ + int((new_y - cy_) * cfg.line_spacing)
                cy_ += cfg.para_spacing
            return boxes_

        yolo_boxes.extend(render_col(col1_sents, col1_x, col_w))
        yolo_boxes.extend(render_col(col2_sents, col2_x, col_w))

        # Advance y to cover full column height (approximate)
        y = min(footer_y0 - 10, y + (cfg.n_paragraphs * sents_per_p
                                      * cfg.body_size * 2))

    else:
        # ── Single-column body ─────────────────────────────────────────────
        # REPORT template: add section-number labels occasionally
        section_labels = random.random() < 0.40 and cfg.template == LayoutTemplate.REPORT
        section_num = 1

        for p_idx in range(cfg.n_paragraphs):
            if y + cfg.body_size >= footer_y0:
                break

            # Optional section heading (REPORT style)
            if section_labels and random.random() < 0.60:
                lbl_font = _resolve_font(cfg, font_files, "header")
                lbl_text = f"{section_num}. {random.choice(DOCUMENT_TITLES)[0]}"
                y, boxes = _draw_section_lines(
                    draw, [lbl_text], cfg.ml, y, lbl_font, cfg.ink,
                    text_w, cfg.page_w, cfg.page_h,
                    align="left", line_spacing=1.0,
                )
                yolo_boxes.extend(boxes)
                section_num += 1
                y += random.randint(4, 10)

            start = p_idx * sents_per_p
            end   = start + sents_per_p + random.randint(0, 2)
            para  = " ".join(all_sents[start:end])
            if not para.strip():
                continue

            wrapped = wrap_text(para, body_font, text_w, draw)
            for li, wl in enumerate(wrapped):
                if y + cfg.body_size >= footer_y0:
                    break
                x_pos = cfg.ml + (cfg.indent if li == 0 else 0)
                new_y, bbox = draw_line(
                    draw, wl, x_pos, y, body_font, cfg.ink,
                    cfg.page_w, cfg.page_h, align="left",
                )
                if bbox:
                    yolo_boxes.append(bbox)
                y = y + int((new_y - y) * cfg.line_spacing)

            y += cfg.para_spacing

    # =====================================================================
    # BLOCK E – Footer / Signature
    # =====================================================================
    if cfg.has_footer:
        foot_font  = _resolve_font(cfg, font_files, "footer")
        foot_entry = random.choice(FOOTER_SIGNATURES)

        # Footer x / width depend on alignment choice
        if cfg.footer_align == "right":
            foot_x = cfg.page_w // 2
            foot_w = cfg.page_w - foot_x - cfg.mr
        elif cfg.footer_align == "center":
            foot_x = cfg.ml
            foot_w = text_w
        else:   # "left"
            foot_x = cfg.ml
            foot_w = text_w // 2

        y_foot = footer_y0
        y_foot, boxes = _draw_section_lines(
            draw, foot_entry, foot_x, y_foot, foot_font, cfg.ink,
            foot_w, cfg.page_w, cfg.page_h,
            align=cfg.footer_align, line_spacing=cfg.line_spacing,
        )
        yolo_boxes.extend(boxes)

    # ── Overlay stamps / logos / non-text objects (unannotated) ─────────────
    # These are pasted BEFORE augmentation so they share the same scan
    # artefacts as the text, making them harder to filter by quality alone.
    if object_files is None:
        object_files = []
    canvas = paste_objects_on_canvas(
        canvas       = canvas,
        object_files = object_files,
        page_w       = cfg.page_w,
        page_h       = cfg.page_h,
        text_boxes   = yolo_boxes,   # prevents objects overlapping text
    )

    # ── Augment ───────────────────────────────────────────────────────────────
    canvas = augment_image(canvas, aug_standard, aug_heavy)

    return canvas, yolo_boxes



# ============================================================================
# SECTION 9 – Output writer
# ============================================================================

def write_outputs(
    image:      Image.Image,
    yolo_boxes: list[tuple[float, float, float, float]],
    stem:       str,
    img_dir:    Path,
    lbl_dir:    Path,
) -> None:
    """Save the image as JPEG and write its YOLO label file."""
    img_path = img_dir / f"{stem}.jpg"
    lbl_path = lbl_dir / f"{stem}.txt"

    image.save(str(img_path), format="JPEG", quality=93)

    with open(lbl_path, "w", encoding="utf-8") as f:
        for cx, cy, w, h in yolo_boxes:
            # class_id = 0 (single class: text-line)
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ============================================================================
# SECTION 10 – dataset.yaml
# ============================================================================

YAML_TEMPLATE = """\
# YOLO dataset configuration – Khmer Document Text-Line Detection
path: {root}
train: images/train
val:   images/val

nc: 1
names:
  0: text_line
"""

def write_dataset_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(YAML_TEMPLATE.format(root=str(output_dir.resolve())))
    print(f"[✓] dataset.yaml written → {yaml_path}")


# ============================================================================
# SECTION 11 – CLI entry point + parallel generation engine
# ============================================================================

import multiprocessing as mp
import os
import time
from multiprocessing import Queue


# ---------------------------------------------------------------------------
# Worker state
#   albumentations Compose objects are not picklable, so each worker builds
#   its own copy after forking.  Font files and corpus strings are plain
#   picklable lists and are passed through the initializer.
# ---------------------------------------------------------------------------
_worker_font_files:   list[Path]  = []
_worker_corpus:       list[str]   = []
_worker_object_files: dict        = {'stamps':[],'logos':[],'seals':[],'misc':[]}
_worker_aug_standard: A.Compose   = None   # type: ignore[assignment]
_worker_aug_heavy:    A.Compose   = None   # type: ignore[assignment]


def _worker_init(font_files: list[Path], corpus_lines: list[str], object_files: list[Path] = None) -> None:
    """
    Pool initializer – called once per worker process.
    Builds the albumentations pipelines locally (not picklable over the queue)
    and stores all shared state in module-level globals.
    """
    global _worker_font_files, _worker_corpus, _worker_object_files
    global _worker_aug_standard, _worker_aug_heavy

    _worker_font_files   = font_files
    _worker_corpus       = corpus_lines
    _worker_object_files = (
        object_files if object_files is not None
        else {'stamps': [], 'logos': [], 'seals': [], 'misc': []}
    )
    _worker_aug_standard, _worker_aug_heavy = build_augmentation_pipeline()

    # Give each worker a unique random seed derived from its PID so documents
    # across workers don't repeat the same random sequence.
    seed = os.getpid() ^ int(time.time() * 1000) & 0xFFFF_FFFF
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)


def _generate_one(task: tuple[int, str, Path, Path]) -> tuple[int, str, int]:
    """
    Worker function: generate one document and write it to disk.

    Parameters
    ----------
    task : (global_index, stem_name, img_dir, lbl_dir)

    Returns
    -------
    (global_index, stem_name, num_yolo_boxes)
        Used by the main process to print progress.
    """
    idx, stem, img_dir, lbl_dir = task
    try:
        cfg   = DocConfig.sample(_worker_font_files)
        image, yolo_boxes = generate_document(
            font_files   = _worker_font_files,
            corpus_lines = _worker_corpus,
            aug_standard = _worker_aug_standard,
            aug_heavy    = _worker_aug_heavy,
            cfg          = cfg,
            object_files = _worker_object_files,
        )
        write_outputs(image, yolo_boxes, stem, img_dir, lbl_dir)
        return idx, stem, len(yolo_boxes)
    except Exception as exc:
        # Workers must never crash – return a sentinel so the main process
        # can log and skip without losing the whole batch.
        return idx, f"ERROR:{stem}:{exc}", -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Cambodian document generator for YOLO text-line detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""            Directory layout:
              ./fonts/                   – .ttf / .otf font files
              ./texts/khmer_corpus.txt   – Khmer text corpus
              ./datasets/images/train/   – output images
              ./datasets/labels/train/   – output YOLO labels

            Speed tips:
              --workers 0   → use all logical CPU cores (default)
              --workers 4   → use exactly 4 cores
              --workers 1   → single-process (debug / reproducibility)
              --chunksize 8 → tune IPC overhead vs. memory; larger = fewer
                              round-trips but more RAM per batch
        """),
    )
    parser.add_argument("--count", "-n", type=int, default=100,
                        help="Number of documents to generate (default: 100).")
    parser.add_argument("--output", "-o", type=str, default="./datasets",
                        help="Root output directory (default: ./datasets).")
    parser.add_argument("--fonts",  type=str, default="./fonts",
                        help="Font directory (default: ./fonts).")
    parser.add_argument("--corpus", type=str, default="./texts/khmer_corpus.txt",
                        help="Path to Khmer text corpus.")
    parser.add_argument("--split",  type=float, default=0.85,
                        help="Train split fraction (rest → val). Default: 0.85.")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for the main process (workers get "
                             "independent PID-derived seeds).")
    parser.add_argument(
        "--workers", "-w", type=int, default=0,
        help="Worker processes. 0 = all CPU cores (default). 1 = single-process.",
    )
    parser.add_argument(
        "--chunksize", type=int, default=4,
        help="Tasks per IPC chunk sent to each worker (default: 4). "
             "Increase if you have many cores and large images.",
    )
    parser.add_argument(
        "--objects", type=str, default="./objects",
        help="Directory containing stamp/logo/non-text overlay images "
             "(default: ./objects). Overlays are pasted WITHOUT annotation "
             "so the model learns to ignore non-text elements. "
             "Skipped silently if the directory does not exist.",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Auto-detect the highest existing kh_doc_NNNNNN index and "
             "continue numbering from there. Use this to top-up an "
             "existing dataset without overwriting any files.",
    )
    return parser.parse_args()


def _find_resume_index(
    train_img_dir: Path,
    val_img_dir:   Path,
) -> int:
    """
    Scan existing output directories for files matching kh_doc_NNNNNN.jpg
    and return (max_index + 1) so generation resumes without overwriting.

    Returns 0 if no existing files are found.
    """
    pat = re.compile(r"kh_doc_(\d{6})\.jpg$")
    max_idx = -1
    for d in [train_img_dir, val_img_dir]:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = pat.match(f.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def main() -> None:
    args = parse_args()

    # Seed (main process only; workers get independent PID-derived seeds)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[i] Main-process seed: {args.seed}")

    # Output directories
    root = Path(args.output)
    n_train = int(args.count * args.split)
    splits = {
        "train": {
            "images": root / "images" / "train",
            "labels": root / "labels" / "train",
        },
        "val": {
            "images": root / "images" / "val",
            "labels": root / "labels" / "val",
        },
    }
    for split_dirs in splits.values():
        for d in split_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # Load assets once in the main process
    font_files   = load_fonts(args.fonts)
    corpus_lines = load_corpus(args.corpus)
    object_files = load_object_pool(args.objects)

    # ── Resume: find highest existing index ───────────────────────────────
    start_idx = 0
    if args.resume:
        start_idx = _find_resume_index(
            splits["train"]["images"],
            splits["val"]["images"],
        )
        if start_idx > 0:
            print(f"[i] Resume mode: continuing from index {start_idx} "
                  f"(skipping {start_idx} already-generated docs)")
        else:
            print("[i] Resume mode: no existing files found, starting from 0")

    n_workers = args.workers if args.workers > 0 else mp.cpu_count()

    print("\n[i] Fonts loaded     :", len(font_files))
    print("[i] Corpus lines     :", len(corpus_lines))
    print("[i] Object assets    :", len(object_files))
    print("[i] Documents to gen :", args.count, "(new)")
    print("[i] Starting index   :", start_idx)
    print("[i] Workers          :", n_workers, "/", mp.cpu_count(), "logical CPUs")
    print("[i] Chunk size       :", args.chunksize)
    print("[i] Output root      :", root.resolve(), "\n")

    # ── Build task list (indices start at start_idx) ─────────────────────
    # n_train / n_val proportions are computed over the NEW docs only;
    # existing files keep whatever split they were originally assigned.
    tasks = []
    for local_i in range(args.count):
        global_i = start_idx + local_i
        split    = "train" if local_i < n_train else "val"
        stem     = f"kh_doc_{global_i:06d}"
        img_dir  = splits[split]["images"]
        lbl_dir  = splits[split]["labels"]
        tasks.append((global_i, stem, img_dir, lbl_dir))

    t0     = time.perf_counter()
    done   = 0
    errors = 0

    def _report(stem_out, n_boxes):
        nonlocal done, errors
        done += 1
        if n_boxes < 0:
            errors += 1
            print(f"  [ERROR] {stem_out}")
        elif done % 10 == 0 or done == args.count:
            elapsed = time.perf_counter() - t0
            rate    = done / max(elapsed, 1e-6)
            eta     = (args.count - done) / max(rate, 1e-6)
            print(
                f"  [{done:>5}/{args.count}]  {stem_out}.jpg"
                f"  ({n_boxes} lines)"
                f"  {rate:.1f} doc/s"
                f"  ETA {eta:.0f}s"
            )

    if n_workers == 1:
        # Single-process path: no spawn overhead, easier to debug
        _worker_init(font_files, corpus_lines, object_files)
        for task in tasks:
            idx, stem_out, n_boxes = _generate_one(task)
            _report(stem_out, n_boxes)
    else:
        # Multi-process path using spawn context (safe on all platforms)
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes   = n_workers,
            initializer = _worker_init,
            initargs    = (font_files, corpus_lines, object_files),  # object_files is dict
        ) as pool:
            for idx, stem_out, n_boxes in pool.imap_unordered(
                _generate_one,
                tasks,
                chunksize=args.chunksize,
            ):
                _report(stem_out, n_boxes)

    write_dataset_yaml(root)

    elapsed = time.perf_counter() - t0
    n_val   = args.count - n_train
    print("\n[OK] Done in", round(elapsed, 1), "s  (", round(args.count / elapsed, 1), "doc/s )")
    print("     Train :", n_train, " | Val :", n_val, " | Errors :", errors)
    print("     Root  :", root.resolve(), "\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Guard required for multiprocessing on Windows / macOS spawn context
    main()