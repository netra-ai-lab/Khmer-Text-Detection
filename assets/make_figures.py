"""README figures in the Netra style.

    python assets/make_figures.py          # writes the PNGs next to this file

training.png: validation accuracy after each epoch of the second training phase. The numbers
are read from assets/train_results_phase2.json, which is the `train_results` log stored inside
src/yolo26s_best/best.pt (torch.load(...)["train_results"]), not typed in by hand.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "brand"))
from brand_colors import COLOR, NEUTRALS, OURS, apply  # noqa: E402

INK = COLOR["foreground"]


def training_curve(log: dict, out: Path) -> None:
    epochs = log["epoch"]
    loose = log["metrics/mAP50(B)"]
    strict = log["metrics/mAP50-95(B)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, loose, color=NEUTRALS[2], linewidth=2, zorder=3)
    ax.plot(epochs, strict, color=OURS, linewidth=2.6, marker="o", markersize=4.5, zorder=4)
    ax.text(epochs[-1] + 0.4, loose[-1], f"{loose[-1]:.3f}\nmAP@0.50", va="center", fontsize=10, color=INK)
    ax.text(epochs[-1] + 0.4, strict[-1], f"{strict[-1]:.3f}\nmAP@0.50:0.95", va="center", fontsize=10,
            color=INK, fontweight="bold")
    ax.text(epochs[0] + 0.3, strict[0] - 0.012, f"{strict[0]:.3f}", va="top", fontsize=10, color=INK)
    ax.set_xlim(0.5, epochs[-1] + 4.6)
    ax.set_ylim(0.70, 1.0)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Epoch of phase 2")
    ax.set_ylabel("Validation score, higher is better")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    apply()
    log = json.loads((HERE / "train_results_phase2.json").read_text(encoding="utf-8"))
    training_curve(log, out_dir / "training.png")


if __name__ == "__main__":
    main(HERE)
