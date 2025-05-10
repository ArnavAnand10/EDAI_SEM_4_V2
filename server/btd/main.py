from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms
from typing import Optional
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch.nn.functional as F

app = FastAPI(title="Medical Deepfake Detection API")

MODALITY_CONFIG = {
    "CT": {
        "patch_size": 96,
        "clip_range": (-700, 2000),
        "normalize_range": (0, 2700)  # (min + 700, max + 700)
    },
    "MRI": {
        "patch_size": 128,
        "clip_range": None,  # No clipping for MRI
        "normalize_range": (0, 1)     # Normalize to [0,1]
    }
}

# Model configuration (from your notebook)
image_size = 128  # For MRI, 96 for CT
channels = 1      # Grayscale medical images

# Initialize the model (same as in your notebook)
model = Unet(
    dim=32,
    dim_mults=(1, 2, 4, 8),
    channels=channels,
    out_dim=channels
)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,   # Number of diffusion steps
    # loss_type='l1'    # L1 or L2
).to('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        # This matches the error you're seeing - the checkpoint contains training state
        model.load_state_dict(checkpoint['model'])
    else:
        # Direct model weights
        model.load_state_dict(checkpoint)
    
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load your pretrained weights here (replace with your actual path)
# diffusion.load_state_dict(torch.load('./weights/CT_model.pt', map_location=torch.device("cpu")))
# diffusion.load_state_dict(torch.load('./weights/MRI_model.pt', map_location=torch.device("cpu")))
diffusion = load_model_weights(diffusion, './weights/CT_model.pt')


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

class PredictionResponse(BaseModel):
    is_fake: bool
    confidence: float
    anomaly_score: float
    modality: str
    message: Optional[str] = None

def preprocess_image(image_data, modality='MRI', patch_size=128, x=256, y=256):
    """Preprocess image based on modality with same logic as notebook"""
    try:
        config = MODALITY_CONFIG[modality]
        patch_size = config["patch_size"]
        
        if modality == 'CT':
            # For CT - assuming numpy array input
            image = np.load(BytesIO(image_data))
            image = np.clip(image, *config["clip_range"])
            image = (image - config["clip_range"][0]) / (config["normalize_range"][1] - config["normalize_range"][0])
            image = image.astype(np.float32)
        else:  # MRI
            image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
            image = np.array(image).astype(np.float32)
            # Normalize to [0,1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Handle vertical flipping as in notebook
            if np.sum(image[-30:]) < np.sum(image[:30]):
                image = np.flip(image, axis=0).copy()
                y = image.shape[0] - y  # Adjust y coordinate if flipped
        
        # Calculate patch coordinates (same logic as notebook)
        half_size = patch_size // 2
        
        # Adjust coordinates if near boundaries
        y = max(half_size, min(y, image.shape[0] - half_size))
        x = max(half_size, min(x, image.shape[1] - half_size))
        
        # Extract patch
        patch = image[y - half_size: y + half_size, x - half_size: x + half_size]
        
        # Ensure patch is exactly the right size (in case of odd dimensions)
        if patch.shape != (patch_size, patch_size):
            patch = patch[:patch_size, :patch_size]
        
        # Reshape to (1, patch_size, patch_size)
        patch = patch.reshape(1, patch_size, patch_size)
        
        return torch.from_numpy(patch).unsqueeze(0).float()  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def detect_anomaly(image_tensor):
    """Run diffusion model to detect anomalies (same approach as notebook)"""
    image_tensor = image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Forward diffusion process
    noise = torch.randn_like(image_tensor)
    t = torch.randint(0, diffusion.num_timesteps, (1,), device=image_tensor.device).long()
    noisy_image = diffusion.q_sample(image_tensor, t, noise)
    
    # Reverse diffusion process
    pred_noise = diffusion.model(noisy_image, t)
    
    # Calculate reconstruction error
    loss = F.l1_loss(pred_noise, noise, reduction='none')
    anomaly_score = loss.mean().item()
    
    # Threshold for fake/real (adjust based on your validation)
    is_fake = anomaly_score > 0.1  # Example threshold
    confidence = min(anomaly_score / 0.2, 1.0)  # Scale to 0-1 range
    
    return is_fake, confidence, anomaly_score

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)