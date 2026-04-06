import argparse
import cv2
import json 
from pathlib import Path
from PIL import Image
from core import DocumentAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Parse documents for textlines and logos.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="yolo26s_best/best.pt")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    # Create output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model... ({args.model})")
    analyzer = DocumentAnalyzer(model_path=args.model)

    print(f"Analyzing {args.image}...")
    img = Image.open(args.image).convert("RGB")
    prediction = analyzer.predict(img)

    # 1. Save the annotated image (the one with boxes drawn on it)
    out_img_path = out_dir / f"annotated_{Path(args.image).name}"
    cv2.imwrite(str(out_img_path), prediction["annotated_image"])
    
    # 2. Save cropped logos/images
    for i, det in enumerate(prediction["detections"]):
        if det["class_id"] == 1: # Class 1 is 'image/logo'
            crop_path = out_dir / f"logo_{i}_{Path(args.image).name}"
            x1, y1, x2, y2 = map(int, det["bbox"])
            cropped_logo = img.crop((x1, y1, x2, y2))
            cropped_logo.save(crop_path)

    # 3. Save the results as a JSON file
    image_stem = Path(args.image).stem  # gets "test_doc" from "test_doc.jpg"
    json_path = out_dir / f"{image_stem}_results.json"
    
    output_data = {
        "filename": Path(args.image).name,
        "total_objects": len(prediction["detections"]),
        "detections": prediction["detections"]
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Done! Found {len(prediction['detections'])} objects.")
    print(f"Results (Images & JSON) saved to {out_dir}/")

if __name__ == "__main__":
    main()

    # python cli.py --image test_doc.jpg --output results/