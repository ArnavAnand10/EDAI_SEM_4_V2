from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io

from pydantic import BaseModel

from mits_gan.service import (
    embed_metadata_in_image,
    extract_metadata_from_image,
    preprocess_image,
    load_model,
)
from btd.services import (
    MODALITY_CONFIG,
    preprocess_image,
    detect_anomaly,
)
app = FastAPI(
    title="Image Forgery Detection API",
    description="API to detect if an image is authentic or forged",
)

model_path = "casia2_model.h5"
model = None


class VerificationResult(BaseModel):
    original_metadata: dict
    is_intact: bool
    message: str


class PredictionResponse(BaseModel):
    is_fake: bool
    confidence: float
    anomaly_score: float
    modality: str
    message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/detect/")
async def detect_forgery(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists.",
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        if h != 256 or w != 384:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions must be 384x256 pixels. Got {w}x{h}.",
            )

        processed_img = preprocess_image(img)

        input_img = np.expand_dims(processed_img, axis=0)

        prediction = model.predict(input_img)
        prediction_class = int(np.round(prediction[0][0]))
        prediction_confidence = float(prediction[0][0])

        result = {
            "is_forged": bool(prediction_class),
            "confidence": prediction_confidence,
            "result": "Forged" if prediction_class == 1 else "Authentic",
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during processing: {str(e)}"
        )
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import imghdr  # For image type detection
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       "*"
        # Add other origins as needed
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Exposes all headers
)

class VerificationResult(BaseModel):
    original_metadata: dict
    is_intact: bool
    message: str

from mits_gan.service import (
    embed_metadata_in_image,
    extract_metadata_from_image,
    preprocess_image,
    load_model,
)


@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await image.read()

        processed_image, metadata = embed_metadata_in_image(image_data)

        return StreamingResponse(
            io.BytesIO(processed_image),
            media_type=image.content_type,
            headers={
                "Content-Disposition": f"attachment; filename=embedded_{image.filename}",
                "X-Metadata-UUID": metadata["uuid"],
                "X-Metadata-Timestamp": str(metadata["timestamp"]),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


    
@app.post("/verify", response_model=VerificationResult)
   
async def verify_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await image.read()

        metadata = extract_metadata_from_image(image_data)

        if metadata:
            return VerificationResult(
                original_metadata=metadata,
                is_intact=True,
                message="Image metadata extracted successfully, image appears intact.",
            )
        else:
            return VerificationResult(
                original_metadata={},
                is_intact=False,
                message="Failed to extract metadata. The image may have been tampered with or does not contain embedded metadata.",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying image: {str(e)}")



@app.post("/predict", response_model=PredictionResponse)
async def predict_deepfake(
    file: UploadFile = File(...),
    modality: str = "MRI", 
    x: int = 256,
    y: int = 256
):
    """Endpoint for medical deepfake detection using diffusion model"""
    try:
        # Validate input
        if modality not in MODALITY_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported modality. Must be one of: {list(MODALITY_CONFIG.keys())}"
            )
        if modality not in ['CT', 'MRI']:
            raise HTTPException(status_code=400, detail="Modality must be either 'CT' or 'MRI'")
        
        # Read file content
        contents = await file.read()
        
        # Preprocess based on modality
        patch_size = 96 if modality == 'CT' else 128
        input_tensor = preprocess_image(contents, modality, patch_size, x, y)
        
        # Run detection
        is_fake, confidence, anomaly_score = detect_anomaly(input_tensor)
        
        return {
            "is_fake": bool(is_fake),
            "confidence": float(confidence),
            "anomaly_score": float(anomaly_score),
            "modality": modality,
            "message": "Prediction successful"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
