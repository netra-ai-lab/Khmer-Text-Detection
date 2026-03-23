import os
import random
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration (Update these if your folder structure is different)
# ---------------------------------------------------------------------------
IMAGE_DIR = 'dataset/images/train'
LABEL_DIR = 'dataset/labels/train'

# The exact 12 classes we defined earlier
CLASSES =[
    "0: id_number", 
    "1: name_kh", 
    "2: name_en", 
    "3: dob_sex_height", 
    "4: pob",
    "5: address_1", 
    "6: address_2", 
    "7: validity", 
    "8: features",
    "9: mrz_1", 
    "10: mrz_2", 
    "11: mrz_3"
]

# Generate random colors for each class to make verification visually easier
random.seed(42) # Fixed seed so colors stay consistent across runs
COLORS =[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(CLASSES))]

def verify_random_sample():
    # 1. Get all images in the directory
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images =[f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    # 2. Pick a random image
    img_filename = random.choice(images)
    img_path = os.path.join(IMAGE_DIR, img_filename)
    
    # 3. Construct matching label filename
    # E.g. "image_001.jpg" -> "image_001.txt"
    base_name = os.path.splitext(img_filename)[0]
    label_filename = f"{base_name}.txt"
    label_path = os.path.join(LABEL_DIR, label_filename)
    
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {img_filename} at {label_path}")
        return

    # 4. Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}")
        return
        
    img_h, img_w, _ = img.shape
    
    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5. Read and parse the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"--- Verifying Sample ---")
    print(f"Image: {img_filename}")
    print(f"Resolution: {img_w}x{img_h}")
    print(f"Total Detections: {len(lines)}\n")

    # 6. Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        # Convert YOLO normalized coordinates (0 to 1) back to absolute pixel values
        x_center = x_center_norm * img_w
        y_center = y_center_norm * img_h
        box_w = width_norm * img_w
        box_h = height_norm * img_h

        # Calculate bounding box corners (x_min, y_min, x_max, y_max)
        x_min = int(x_center - (box_w / 2))
        y_min = int(y_center - (box_h / 2))
        x_max = int(x_center + (box_w / 2))
        y_max = int(y_center + (box_h / 2))

        # Get color and class name
        color = COLORS[class_id]
        label_name = CLASSES[class_id]

        # Draw Rectangle
        cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw Background for Text (makes it easier to read)
        (text_w, text_h), baseline = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_rgb, (x_min, y_min - text_h - 4), (x_min + text_w, y_min), color, -1)
        
        # Put Class Name Text
        cv2.putText(img_rgb, label_name, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 7. Save and Display
    output_filename = 'verify_sample_output.jpg'
    # Convert back to BGR for saving with cv2
    cv2.imwrite(output_filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved verification image to: {output_filename}")

    # Display using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Ground Truth Check: {img_filename}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_random_sample()