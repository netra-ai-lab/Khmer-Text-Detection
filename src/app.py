import streamlit as st
from PIL import Image
from core import DocumentAnalyzer

st.set_page_config(page_title="Document Analyzer", layout="wide")

# Cache the model
@st.cache_resource
def load_model():
    return DocumentAnalyzer()

analyzer = load_model()

# 👇 Added a Sidebar for Model Settings
st.sidebar.title("⚙️ Model Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05, 
    max_value=0.95, 
    value=0.45, 
    step=0.05,
    help="Increase this to filter out low-confidence detections. Decrease to catch hard-to-read text."
)

st.title("📄 Multilingual Textline & Logo Extractor")
st.markdown("Upload a document, ID card, or scene image to detect text and extract graphical elements.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run Inference using the slider value!
    with st.spinner("Analyzing document..."):
        prediction = analyzer.predict(image, conf_threshold=conf_threshold)
        
    st.success(f"Found {len(prediction['detections'])} objects at ≥ {conf_threshold} confidence!")

    # UI Layout: 2 Columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Annotated Document")
        # Convert BGR back to RGB for Streamlit
        rgb_annotated = prediction["annotated_image"][..., ::-1] 
        st.image(rgb_annotated, use_container_width=True)

    with col2:
        st.subheader("Extracted Assets")
        logos_found = 0
        for det in prediction["detections"]:
            if det["class_id"] == 1: # Image/Logo class
                logos_found += 1
                x1, y1, x2, y2 = map(int, det["bbox"])
                st.image(image.crop((x1, y1, x2, y2)), caption=f"Confidence: {det['confidence']:.2f}")
                
        if logos_found == 0:
            st.info("No logos or images found in this document.")
            
        st.subheader("Raw JSON Output")
        
        # Wrapped the output to include the threshold used
        st.json({
            "confidence_threshold": conf_threshold,
            "total_objects": len(prediction["detections"]),
            "detections": prediction["detections"]
        })