# 🔍 Multilingual Textline & Logo Detection (YOLO26s)

![YOLO26s](https://img.shields.io/badge/Model-YOLO26s-blue) ![Dataset](https://img.shields.io/badge/Dataset-~30K%20Images-green) ![Task](https://img.shields.io/badge/Task-Object%20Detection-orange) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 📖 Project Overview
This repository contains the training pipeline and methodology for a highly robust, dual-class object detection model designed to bridge the gap between document analysis and scene text detection. 

The primary objective is to accurately localize **tightly packed text lines** (specifically optimized for complex scripts like Khmer, alongside Latin and Russian) and **graphical elements** (logos, profile pictures, and charts). The model is designed to be highly generalized, performing equally well on clean synthetic ID cards, noisy scanned documents, and natural scene text "in the wild."

---

## ✨ Model Capabilities
*   **Target Classes:** 
    *   `0: textline` (Lines of text, independent of language)
    *   `1: image` (Logos, graphical charts, profile pictures)
*   **Language Agnostic:** Learns structural feature representations of textlines rather than character recognition. Highly optimized for **Latin, Russian, and Khmer**.
*   **Core Use-Cases:** 
    *   **Pre-OCR Pipeline:** Accurately mapping textlines to feed into downstream Optical Character Recognition (OCR) engines.
    *   **Entity Extraction:** Cropping profile pictures from ID cards or isolating logos/charts from dense PDFs.

---

## 🏗️ Project Architecture
The project transitioned from an initial baseline architecture to the state-of-the-art **YOLO26s** to solve specific challenges inherent to dense text detection. 

*   **NMS-Free Architecture:** Leverages YOLO26's end-to-end design, eliminating Non-Maximum Suppression (NMS) post-processing. This prevents the model from accidentally suppressing overlapping or tightly stacked lines of text.
*   **STAL (Small-Target-Aware Label Assignment):** Utilizes YOLO26's native STAL to vastly improve the recall and precision of extremely small, dense textlines.
*   **Optimizer:** `MuSGD` for highly stable convergence during multi-domain fine-tuning.

---

## 🗄️ Dataset Pipeline
Building a model capable of handling multiple domains required a hybrid approach, merging custom synthetic generation with large-scale open-source datasets via the Hugging Face Hub.

### 1. Custom Synthetic Generation
To ensure the model understood specific layout constraints (like Cambodian ID cards), we built a programmatic synthetic data engine:
*   **Procedural Generation:** Python scripts generated thousands of randomized document pages and ID cards.
*   **Typographical Variance:** Loaded custom font files to support complex Khmer ligatures alongside standard Latin/Cyrillic fonts.
*   **Asset Injection:** Programmatically injected logos, profile photos, and graphical elements with mathematically perfect ground-truth bounding boxes.

### 2. External Data Integration
To ensure real-world robustness and prevent overfitting to synthetic clean data, we streamed external datasets:
*   **DonkeySmall:** For natural scene text in Latin and Russian.
*   **DocLayNet (IBM):** For dense, complex, real-world document layouts with high-quality bounding boxes for charts and pictures.
*   **Khmer Textline Dataset:** For real-world, noisy Cambodian text structures.

---

## 🧠 Training Strategy & Data Engineering

### 🚨 The Challenge: "Missing Label Poisoning"
When merging the datasets, a critical data-science hurdle arose: our external textline datasets (and some early synthetic data) *only* contained annotations for text, completely ignoring the logos and images present in the pixels. 

Training a multi-class model on this data directly would cause **False Negative Poisoning** (or Missing Label Poisoning). The model's loss function would penalize it for detecting a logo because the ground-truth file claimed it was "background noise." This would destroy the model's ability to reliably detect the `image` class.

### 🛠️ The Solution: Bootstrap Pseudo-Labeling
To protect the dataset integrity without spending hundreds of hours on manual annotation, we implemented an automated pseudo-labeling pipeline:

1.  **Baseline Training:** Trained a baseline textline-only detector (using YOLOv11s) on our text-only datasets.
2.  **Cross-Dataset Inference:** Ran the baseline model over the IBM DocLayNet dataset to auto-annotate all `textline` instances (Class 0), while extracting IBM's highly accurate ground-truth coordinates for the `image/picture` class (Class 1).
3.  **Auto-Annotation:** Using this newly balanced 2-class model, we ran inference *back* over our unannotated 2,000+ custom Khmer images to automatically draw the missing logo bounding boxes.

**Result:** A perfectly unified, mathematically accurate multi-class dataset where *every single image* was correctly annotated for both textlines and images, completely eliminating data poisoning.

### 🚀 Final Mega-Training
With the dataset fully cleaned, balanced, and merged into a single `train/val` pipeline (~30,000 images), the final training phase was executed:
*   Upgraded to **YOLO26s** pretrained weights.
*   Concatenated the Synthetic ID Cards, Synthetic Documents, DonkeySmall Scene Text, DocLayNet, and the custom auto-annotated Khmer data.
*   Trained simultaneously on the combined mega-dataset to ensure a highly generalized, unified feature-extraction head.

---

## 💻 Getting Started (Inference Example)

```python
from ultralytics import YOLO
from PIL import Image

# Load the trained YOLO26s model
model = YOLO("weights/yolo26s_ultimate_detector.pt")

# Run inference on a document or ID card
results = model.predict("sample_document.jpg", conf=0.4)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    label = model.names[class_id]
    print(f"Detected: {label} at {box.xyxy[0]}")
    