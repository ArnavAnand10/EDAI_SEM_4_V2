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

app = FastAPI()

def embed_metadata_in_image(img):
    # Assuming this function exists elsewhere in your code
    # and returns a processed image and metadata
    metadata = {"embedded": True}
    return img, metadata

@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
    try:
        # 1. Read input file
        image_data = await image.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # 2. Detect image format for debugging
        image_format = imghdr.what(None, h=image_data)
        if not image_format:
            raise HTTPException(status_code=400, detail="Could not identify image format")
        
        # 3. Create a file-like object for OpenCV
        image_stream = io.BytesIO(image_data)
        image_stream.seek(0)
        
        # 4. Alternative approach using PIL first to ensure format compatibility
        from PIL import Image
        try:
            pil_image = Image.open(image_stream)
            pil_image.verify()  # Verify the file is a valid image
            image_stream.seek(0)  # Reset stream position
            pil_image = Image.open(image_stream)  # Reopen (verify closes the file)
            
            # Convert PIL image to numpy array for OpenCV
            img = np.array(pil_image.convert('RGB'))
            
            # Convert BGR for OpenCV if needed
            if img.shape[2] == 3:  # Check if it's a 3-channel image
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as pil_error:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image format: {str(pil_error)}, detected format: {image_format}"
            )
        
        # 5. Ensure proper numeric range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 6. Process with error handling
        processed_img, metadata = embed_metadata_in_image(img)
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
        
        # 7. Encode and return
        success, encoded_img = cv2.imencode('.png', processed_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")
            
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()),
            media_type="image/png",
            headers={"X-Metadata": str(metadata)}
        )
        
    except Exception as e:
        # More detailed error reporting
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing image: {error_details}")
        
        # Try to provide more specific error info about the file
        file_info = f"File format detection: {imghdr.what(None, h=image_data) if 'image_data' in locals() else 'unknown'}"
        print(file_info)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}. {file_info}"
        )

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

