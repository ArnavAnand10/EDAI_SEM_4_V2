import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms
from typing import Optional
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch.nn.functional as F


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

# # Model configuration (from your notebook)
# image_size = 128  # For MRI, 96 for CT
# channels = 1      # Grayscale medical images

# # Initialize with EXACTLY the same architecture used to create the checkpoint


# # Initialize with only supported parameters
# model = Unet(
#     dim=32,
#     dim_mults=(1, 2, 4, 8),
#     channels=1,
#     out_dim=1
# )

# # Initialize diffusion model
# diffusion = GaussianDiffusion(
#     model,
#     image_size=128,
#     timesteps=1000
# ).to('cuda' if torch.cuda.is_available() else 'cpu')


# def load_model_with_fallback(model, checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     state_dict = checkpoint.get('model', checkpoint)
    
#     # Create filtered state dict
#     model_dict = model.state_dict()
#     filtered_dict = {}
    
#     # 1. First try direct loading
#     for k, v in state_dict.items():
#         if k in model_dict:
#             filtered_dict[k] = v
    
#     # 2. Try common key variations if direct loading fails
#     if len(filtered_dict) < 0.5 * len(model_dict):  # If less than 50% loaded
#         for k, v in state_dict.items():
#             # Handle common attention key variations
#             new_key = k.replace('fn.fn.to_qkv', 'attn.to_qkv')
#             new_key = new_key.replace('fn.fn.to_out', 'attn.to_out')
            
#             if new_key in model_dict:
#                 filtered_dict[new_key] = v
#             elif k in model_dict:
#                 filtered_dict[k] = v
    
#     # Load whatever weights we can match
#     model.load_state_dict(filtered_dict, strict=False)
    
#     # Print loading statistics
#     print(f"Successfully loaded {len(filtered_dict)}/{len(model_dict)} parameters")
#     missing = set(model_dict.keys()) - set(filtered_dict.keys())
#     if missing:
#         print(f"Missing parameters: {missing}")
    
#     return model
# # Load weights with compatibility handling

# # Initialize diffusion model
# diffusion = GaussianDiffusion(
#     model,
#     image_size=128,
#     timesteps=1000
# ).to('cuda' if torch.cuda.is_available() else 'cpu')

# # Load weights with compatibility handling
# try:
#     diffusion = load_model_with_fallback(diffusion, './btd/weights/CT_model.pt')
# except Exception as e:
#     print(f"Error loading weights: {e}")
#     print("Continuing with randomly initialized weights")

# def load_model_weights(model, checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     # Handle different checkpoint formats
#     if 'model' in checkpoint:
#         # This matches the error you're seeing - the checkpoint contains training state
#         model.load_state_dict(checkpoint['model'])
#     else:
#         # Direct model weights
#         model.load_state_dict(checkpoint)
    
#     return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # Load your pretrained weights here (replace with your actual path)
# # diffusion.load_state_dict(torch.load('./weights/CT_model.pt', map_location=torch.device("cpu")))
# # diffusion.load_state_dict(torch.load('./weights/MRI_model.pt', map_location=torch.device("cpu")))
# diffusion = load_model_weights(diffusion, './btd/weights/CT_model.pt')


# # Define transforms
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# def preprocess_image(image_data, modality='MRI', patch_size=128, x=256, y=256):
#     """Preprocess image based on modality with same logic as notebook"""
#     try:
#         config = MODALITY_CONFIG[modality]
#         patch_size = config["patch_size"]
        
#         if modality == 'CT':
#             # For CT - assuming numpy array input
#             image = np.load(BytesIO(image_data))
#             image = np.clip(image, *config["clip_range"])
#             image = (image - config["clip_range"][0]) / (config["normalize_range"][1] - config["normalize_range"][0])
#             image = image.astype(np.float32)
#         else:  # MRI
#             image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
#             image = np.array(image).astype(np.float32)
#             # Normalize to [0,1]
#             image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
#             # Handle vertical flipping as in notebook
#             if np.sum(image[-30:]) < np.sum(image[:30]):
#                 image = np.flip(image, axis=0).copy()
#                 y = image.shape[0] - y  # Adjust y coordinate if flipped
        
#         # Calculate patch coordinates (same logic as notebook)
#         half_size = patch_size // 2
        
#         # Adjust coordinates if near boundaries
#         y = max(half_size, min(y, image.shape[0] - half_size))
#         x = max(half_size, min(x, image.shape[1] - half_size))
        
#         # Extract patch
#         patch = image[y - half_size: y + half_size, x - half_size: x + half_size]
        
#         # Ensure patch is exactly the right size (in case of odd dimensions)
#         if patch.shape != (patch_size, patch_size):
#             patch = patch[:patch_size, :patch_size]
        
#         # Reshape to (1, patch_size, patch_size)
#         patch = patch.reshape(1, patch_size, patch_size)
        
#         return torch.from_numpy(patch).unsqueeze(0).float()  # Add batch dimension
#     except Exception as e:
#         raise ValueError(f"Image preprocessing failed: {str(e)}")

# def detect_anomaly(image_tensor):
#     """Run diffusion model to detect anomalies (same approach as notebook)"""
#     image_tensor = image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Forward diffusion process
#     noise = torch.randn_like(image_tensor)
#     t = torch.randint(0, diffusion.num_timesteps, (1,), device=image_tensor.device).long()
#     noisy_image = diffusion.q_sample(image_tensor, t, noise)
    
#     # Reverse diffusion process
#     pred_noise = diffusion.model(noisy_image, t)
    
#     # Calculate reconstruction error
#     loss = F.l1_loss(pred_noise, noise, reduction='none')
#     anomaly_score = loss.mean().item()
    
#     # Threshold for fake/real (adjust based on your validation)
#     is_fake = anomaly_score > 0.1  # Example threshold
#     confidence = min(anomaly_score / 0.2, 1.0)  # Scale to 0-1 range
    
#     return is_fake, confidence, anomaly_score

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

def load_model_with_multiple_checkpoints(model, checkpoint_paths):
    """
    Load model weights from multiple checkpoint files with intelligent key remapping.
    
    This function can handle:
    1. Multiple checkpoint files
    2. Architectural differences between checkpoints and the model
    3. Overlapping keys across checkpoints (later checkpoints take precedence)
    
    Args:
        model: The PyTorch model to load weights into
        checkpoint_paths: List of paths to checkpoint files
        
    Returns:
        The model with loaded weights
    """
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]  # Convert single path to list
    
    # Get current model state dict
    model_dict = model.state_dict()
    
    # Track our overall progress
    loaded_keys = set()
    new_state_dict = {}
    
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\nLoading checkpoint {checkpoint_idx+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            continue
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Initialize counters for this checkpoint
        checkpoint_loaded = 0
        
        # First, try direct matching
        for model_key in model_dict.keys():
            # Skip keys we've already loaded from previous checkpoints
            if model_key in loaded_keys:
                continue
                
            if model_key in state_dict:
                if state_dict[model_key].shape == model_dict[model_key].shape:
                    new_state_dict[model_key] = state_dict[model_key]
                    loaded_keys.add(model_key)
                    checkpoint_loaded += 1
        
        # If direct matching is poor, try pattern discovery
        if checkpoint_loaded < 0.5 * len(model_dict):
            print("  Direct matching insufficient, attempting pattern discovery...")
            
            # Analyze key structures to find patterns
            model_patterns = {}
            checkpoint_patterns = {}
            
            # Group keys by their structural patterns
            for key in model_dict.keys():
                if key in loaded_keys:
                    continue
                pattern = ''.join(['N' if c.isdigit() else c for c in key])
                if pattern not in model_patterns:
                    model_patterns[pattern] = []
                model_patterns[pattern].append(key)
            
            for key in state_dict.keys():
                pattern = ''.join(['N' if c.isdigit() else c for c in key])
                if pattern not in checkpoint_patterns:
                    checkpoint_patterns[pattern] = []
                checkpoint_patterns[pattern].append(key)
            
            # Find pattern mappings based on frequency and structure
            pattern_mappings = {}
            for model_pattern, model_keys in model_patterns.items():
                best_match = None
                best_score = 0
                
                for checkpoint_pattern, checkpoint_keys in checkpoint_patterns.items():
                    structure_sim = sum(1 for a, b in zip(model_pattern, checkpoint_pattern) if a == b) / max(len(model_pattern), len(checkpoint_pattern))
                    count_sim = min(len(model_keys), len(checkpoint_keys)) / max(len(model_keys), len(checkpoint_keys))
                    score = structure_sim * 0.7 + count_sim * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_match = checkpoint_pattern
                
                if best_score > 0.6:  # Threshold for considering a good match
                    pattern_mappings[model_pattern] = best_match
            
            # Apply pattern mappings to create key mappings
            pattern_loaded = 0
            for model_pattern, checkpoint_pattern in pattern_mappings.items():
                model_keys = [k for k in model_patterns[model_pattern] if k not in loaded_keys]
                checkpoint_keys = checkpoint_patterns[checkpoint_pattern]
                
                # For simplicity, map by index for equal length groups
                if len(model_keys) == len(checkpoint_keys):
                    model_keys.sort()
                    checkpoint_keys.sort()
                    
                    for i, model_key in enumerate(model_keys):
                        checkpoint_key = checkpoint_keys[i]
                        if state_dict[checkpoint_key].shape == model_dict[model_key].shape:
                            new_state_dict[model_key] = state_dict[checkpoint_key]
                            loaded_keys.add(model_key)
                            pattern_loaded += 1
                
                # For unequal length, try to match by extracting indices
                else:
                    for model_key in model_keys:
                        if model_key in loaded_keys:
                            continue
                            
                        # Extract sequence of digits
                        model_indices = [int(''.join(digit for digit in segment if digit.isdigit())) 
                                         for segment in model_key.split('.') if any(c.isdigit() for c in segment)]
                        
                        best_checkpoint_key = None
                        best_match_count = -1
                        
                        for checkpoint_key in checkpoint_keys:
                            checkpoint_indices = [int(''.join(digit for digit in segment if digit.isdigit())) 
                                                for segment in checkpoint_key.split('.') if any(c.isdigit() for c in segment)]
                            
                            # Count matching indices
                            match_count = sum(1 for a, b in zip(model_indices, checkpoint_indices) if a == b)
                            
                            if match_count > best_match_count:
                                best_match_count = match_count
                                best_checkpoint_key = checkpoint_key
                        
                        if best_checkpoint_key and state_dict[best_checkpoint_key].shape == model_dict[model_key].shape:
                            new_state_dict[model_key] = state_dict[best_checkpoint_key]
                            loaded_keys.add(model_key)
                            pattern_loaded += 1
            
            print(f"  Pattern matching found {pattern_loaded} additional parameters")
            checkpoint_loaded += pattern_loaded
        
        # Apply specific key transformations for known architecture differences
        transform_loaded = 0
        missing_keys = [k for k in model_dict.keys() if k not in loaded_keys]
        
        # Define transformation patterns (model_suffix -> checkpoint_suffix)
        transform_patterns = [
            (".fn.fn.to_qkv.weight", ".to_qkv.weight"),
            (".fn.fn.to_out.0.weight", ".to_out.0.weight"),
            (".fn.fn.to_out.0.bias", ".to_out.0.bias"),
            (".fn.fn.to_out.1.g", ".to_out.1.g"),
            (".fn.norm.g", ".norm.g"),
            (".fn.fn.to_out.weight", ".to_out.weight"),
            (".fn.fn.to_out.bias", ".to_out.bias")
        ]
        
        for missing_key in missing_keys:
            for model_suffix, checkpoint_suffix in transform_patterns:
                if model_suffix in missing_key:
                    checkpoint_key = missing_key.replace(model_suffix, checkpoint_suffix)
                    if checkpoint_key in state_dict and state_dict[checkpoint_key].shape == model_dict[missing_key].shape:
                        new_state_dict[missing_key] = state_dict[checkpoint_key]
                        loaded_keys.add(missing_key)
                        transform_loaded += 1
                        break
        
        print(f"  Transformation rules found {transform_loaded} additional parameters")
        checkpoint_loaded += transform_loaded
        
        # Report checkpoint statistics
        print(f"  Loaded {checkpoint_loaded} parameters from checkpoint {checkpoint_idx+1}")
    
    # Report overall loading statistics
    num_matched = len(loaded_keys)
    num_total = len(model_dict)
    print(f"\nTotal: Successfully loaded {num_matched}/{num_total} parameters ({num_matched/num_total*100:.2f}%)")
    
    # Load the remapped weights
    model.load_state_dict(new_state_dict, strict=False)
    
    # Report remaining missing keys
    still_missing = set(model_dict.keys()) - loaded_keys
    if still_missing:
        print(f"Still missing {len(still_missing)}/{len(model_dict)} keys after loading all checkpoints")
        print(f"Example missing keys: {list(still_missing)[:5]}")
    
    return model


# Load your pretrained weights here (replace with your actual path)
# diffusion.load_state_dict(torch.load('./weights/CT_model.pt', map_location=torch.device("cpu")))
# diffusion.load_state_dict(torch.load('./weights/MRI_model.pt', map_location=torch.device("cpu")))
diffusion = load_model_with_multiple_checkpoints(diffusion, ['CT_model.pt', "MRI_model.pt"])


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor()
])


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
