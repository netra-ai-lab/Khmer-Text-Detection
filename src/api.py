from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from core import DocumentAnalyzer

app = FastAPI(title="Multilingual Document Analyzer API")

# Initialize model globally so it stays in GPU memory between requests
analyzer = DocumentAnalyzer()

@app.post("/predict")
async def predict_document(file: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference
        prediction = analyzer.predict(img)
        
        # We only return the JSON data via API, not the plotted image array
        return JSONResponse(content={
            "filename": file.filename,
            "total_objects": len(prediction["detections"]),
            "detections": prediction["detections"]
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run instruction inside the script for convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)