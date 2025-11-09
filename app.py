"""
FastAPI backend for DFU (Diabetic Foot Ulcer) classification using EfficientNet model.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import io
import base64
import logging
import numpy as np
import cv2
from skimage.filters import threshold_otsu

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DFU Classification API",
    description="API for diabetic foot ulcer classification using EfficientNet-B0",
    version="1.0.0"
)

# Add CORS middleware to allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    # Do not allow credentials when using wildcard origins — browsers disallow
    # Access-Control-Allow-Origin: * together with Access-Control-Allow-Credentials: true.
    # Set to False because the frontend doesn't use cookies/auth credentials.
    allow_credentials=False,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variable to store the model
model = None


# === Grad-CAM Implementation ===
class GradCAM:
    """Grad-CAM implementation for model interpretability."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __enter__(self):
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.forward_handle.remove()
        self.backward_handle.remove()


def compute_grad_cam(model, input_tensor, target_layer, target_class=None):
    """Compute Grad-CAM heatmap."""
    with GradCAM(model, target_layer) as cam:
        model.eval()
        input_tensor.requires_grad_(True)
        output = model(input_tensor)

        if target_class is None:
            target_class = 1 if torch.sigmoid(output).item() >= 0.5 else 0

        model.zero_grad()
        target = torch.ones_like(output) if target_class == 1 else -torch.ones_like(output)
        output.backward(gradient=target, retain_graph=True)

        weights = torch.mean(cam.gradients, dim=(2, 3), keepdim=True)
        cam_map = torch.sum(weights * cam.activations, dim=1, keepdim=True)
        cam_map = F.relu(cam_map).squeeze().cpu().numpy()

        cam_min, cam_max = np.min(cam_map), np.max(cam_map)
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam_map)
        return (cam_map - cam_min) / (cam_max - cam_min + 1e-8)


def overlay_heatmap(img, heatmap, alpha=0.5):
    """Overlay heatmap on image."""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_float = img.astype(np.float32) / 255.0
    heatmap_colored_float = heatmap_colored.astype(np.float32) / 255.0
    blended = cv2.addWeighted(img_float, alpha, heatmap_colored_float, 1 - alpha, 0)
    return np.uint8(255 * blended)


def get_pseudo_segmentation_mask(grayscale_cam, original_image_shape, threshold_method='otsu'):
    """Generate pseudo-segmentation mask from Grad-CAM."""
    cam_resized = cv2.resize(grayscale_cam, original_image_shape)
    if np.max(cam_resized) - np.min(cam_resized) < 1e-6:
        return np.zeros_like(cam_resized, dtype=np.uint8)
    if threshold_method == 'otsu':
        thresh = threshold_otsu(cam_resized)
        binary_mask = cam_resized >= thresh
    else:
        binary_mask = cam_resized >= 0.5
    return binary_mask.astype(np.uint8)


def apply_mask_overlay(image, mask):
    """Apply green overlay on masked regions."""
    green_overlay = image.astype(np.float32) / 255.0
    mask_rgb = np.zeros_like(green_overlay)
    mask_rgb[mask == 1] = [0, 1, 0]  # Green
    blended = cv2.addWeighted(green_overlay, 0.7, mask_rgb, 0.3, 0)
    return np.uint8(255 * blended)


def load_model():
    """Load the PyTorch model at startup."""
    try:
        logger.info("Loading EfficientNet model...")
        
        # Load EfficientNet-B0 architecture
        efficientnet_model = models.efficientnet_b0(weights=None)
        
        # Modify the classifier to match your model's output classes
        # Change num_classes to 1 for binary classification (your model)
        num_features = efficientnet_model.classifier[1].in_features
        efficientnet_model.classifier[1] = nn.Linear(num_features, 1)  # Binary classification
        
        # Load the trained weights
        state_dict = torch.load(
            "efficientnet_dfu_best.pth",
            map_location=torch.device('cpu')
        )
        efficientnet_model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        efficientnet_model.eval()
        
        logger.info("Model loaded successfully!")
        return efficientnet_model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Load model when the application starts."""
    global model
    model = load_model()


@app.get("/")
async def root():
    """Root endpoint for testing if the API is running."""
    return {"message": "DFU API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict endpoint that accepts an image upload and returns prediction.
    
    Args:
        file: Uploaded image file (multipart/form-data)
        
    Returns:
        JSON response with prediction or error message
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )
        
        # Read the uploaded file
        contents = await file.read()
        
        # Open image with PIL
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB (in case image is RGBA or grayscale)
        image = image.convert('RGB')
        
        # Store original size for segmentation
        original_size = (image.width, image.height)
        
        # Resize to 224x224
        image_resized = image.resize((224, 224))
        
        # Convert PIL to numpy for OpenCV operations
        image_np = np.array(image_resized)
        
        # Define the transformation pipeline for model inference
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standard normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transformations
        image_tensor = transform(image_resized)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            prediction_prob = torch.sigmoid(output).item()
            # Class 0 = Abnormal(Ulcer), Class 1 = Normal
            # Lower probability = Abnormal, Higher probability = Normal
            prediction = 0 if prediction_prob < 0.5 else 1
        
        # === Generate Grad-CAM ===
        target_layer = model.features[-1]  # Last convolutional layer
        # For Grad-CAM, use the predicted class
        grayscale_cam = compute_grad_cam(model, image_tensor, target_layer, prediction)
        
        # Create heatmap overlay
        cam_overlay = overlay_heatmap(image_np, grayscale_cam)
        
        # === Generate Pseudo-Segmentation Mask ===
        pseudo_mask = get_pseudo_segmentation_mask(grayscale_cam, (224, 224), 'otsu')
        
        # Calculate ulcer area percentage
        ulcer_area_pct = 0.0
        mask_overlay = image_np.copy()
        
        if prediction == 0:  # Abnormal (Ulcer detected)
            ulcer_pixels = np.sum(pseudo_mask)
            total_pixels = pseudo_mask.shape[0] * pseudo_mask.shape[1]
            ulcer_area_pct = (ulcer_pixels / total_pixels) * 100
            mask_overlay = apply_mask_overlay(image_np, pseudo_mask)
        
        # === Convert all images to base64 ===
        def img_to_base64(img_array):
            """Convert numpy array to base64 string."""
            if img_array.dtype != np.uint8:
                img_array = np.uint8(img_array)
            img_pil = Image.fromarray(img_array)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        original_base64 = img_to_base64(image_np)
        gradcam_base64 = img_to_base64(cam_overlay)
        segmentation_base64 = img_to_base64(mask_overlay)
        
        logger.info(f"Prediction: {prediction}, Probability: {prediction_prob:.4f}, Ulcer Area: {ulcer_area_pct:.2f}%")
        
        # Class names matching training
        class_names = {
            0: "Abnormal(Ulcer)",
            1: "Normal(Healthy skin)"
        }
        
        return {
            "prediction": prediction,
            "probability": round(prediction_prob, 4),
            "class_name": class_names[prediction],
            "ulcer_area_percentage": round(ulcer_area_pct, 2),
            "original_image": f"data:image/jpeg;base64,{original_base64}",
            "gradcam_overlay": f"data:image/jpeg;base64,{gradcam_base64}",
            "segmentation_mask": f"data:image/jpeg;base64,{segmentation_base64}",
            "image_size": "224x224"
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
