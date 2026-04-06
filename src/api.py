from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from core import DocumentAnalyzer

app = FastAPI(title="Multilingual Document Analyzer API")

# Initialize model globally
analyzer = DocumentAnalyzer()

@app.post("/predict")
async def predict_document(
    file: UploadFile = File(...), 
    conf: float = Form(0.45, description="Confidence threshold (0.0 to 1.0)") 
):
    try:
        # Read the uploaded image bytes
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference with the requested confidence
        prediction = analyzer.predict(img, conf_threshold=conf)
        
        return JSONResponse(content={
            "filename": file.filename,
            "confidence_threshold": conf, 
            "total_objects": len(prediction["detections"]),
            "detections": prediction["detections"]
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)