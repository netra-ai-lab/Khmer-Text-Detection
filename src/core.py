import os
from ultralytics import YOLO
from PIL import Image
import io

# 1. Get the exact directory where core.py lives
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the absolute path to your model 
# (This assumes yolo26s_best is sitting right next to core.py in the src folder)
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "yolo26s_best", "best.pt")

class DocumentAnalyzer:
    # 3. Use the dynamic path as the default!
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def predict(self, image: Image.Image, conf_threshold=0.45):
        """Runs YOLO inference and parses the results into a clean dictionary."""
        # YOLO can take PIL images directly
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        result = results[0]
        
        detections =[]
        for box in result.boxes:
            class_id = int(box.cls[0])
            coords = box.xyxy[0].tolist() # [x_min, y_min, x_max, y_max]
            
            detections.append({
                "class_id": class_id,
                "label": self.names[class_id],
                "confidence": float(box.conf[0]),
                "bbox":[round(x, 2) for x in coords]
            })
            
        # result.plot() returns a BGR numpy array with the bounding boxes drawn on it
        annotated_img_array = result.plot()
        
        return {
            "detections": detections,
            "annotated_image": annotated_img_array, # For Streamlit/CLI to show/save
            "raw_result": result                    # For advanced cropping
        }