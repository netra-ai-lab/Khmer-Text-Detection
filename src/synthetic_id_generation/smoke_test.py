"""
smoke_test.py
=============
Runs the generator for 3 cards with NO external assets (no backgrounds, no
custom fonts) and prints every annotation so you can verify the output format
without setting up the full directory tree.

Usage:
    python smoke_test.py
"""
import sys, random, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── make sure the main script is importable from the same directory ──
sys.path.insert(0, str(Path(__file__).parent))
import src.synthetic_id_generation.generate_synthetic_id as G

# ── deterministic output ──
random.seed(42)
np.random.seed(42)

print("=" * 60)
print("Smoke test – 3 synthetic Khmer ID cards (no external assets)")
print("=" * 60)

bg_files  = []                              # fallback to plain colour
font_pool = {"khmer": [], "english": [], "mrz": []}  # fallback to Pillow default
aug       = G.build_augmentation_pipeline()

for card_idx in range(3):
    card, annotations = G.generate_card(bg_files, font_pool, aug, name_idx=card_idx)
    print(f"\n── Card {card_idx} ──  size={card.size}")
    print(f"  {'cls':>3}  {'cx':>8}  {'cy':>8}  {'w':>8}  {'h':>8}  field")
    class_names = [
        "id_number","name_kh","name_en","dob_sex_height",
        "pob","address_1","address_2","validity",
        "features","mrz_1","mrz_2","mrz_3",
    ]
    for (cls_id, cx, cy, w, h) in sorted(annotations, key=lambda x: x[0]):
        print(f"  {cls_id:>3}  {cx:>8.4f}  {cy:>8.4f}  {w:>8.4f}  {h:>8.4f}  {class_names[cls_id]}")

print("\n[✓] Smoke test passed.\n")