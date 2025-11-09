# DFU Classification API

FastAPI backend for Diabetic Foot Ulcer (DFU) classification using EfficientNet-B0 model.

## Features

- 🚀 Fast and efficient REST API using FastAPI
- 🤖 PyTorch EfficientNet-B0 model for image classification
- 📦 Ready for deployment on Render
- 🔍 Health check endpoint
- 📝 Comprehensive error handling

## Project Structure

```
DFU/
├── app.py                          # Main FastAPI application
├── efficientnet_dfu_best.pth       # Trained PyTorch model weights
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment configuration
├── training_history_efficientnet.json  # Training history
└── README.md                       # This file
```

## Installation

### Local Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:\Users\gioan\Desktop\DFU
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Running the API

### Local Development

Run the server locally:
```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs (Swagger UI):** http://localhost:8000/docs
- **Alternative API docs (ReDoc):** http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Test if the API is running.

**Response:**
```json
{
  "message": "DFU API is running"
}
```

### 2. Health Check
**GET** `/health`

Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Predict Endpoint
**POST** `/predict`

Upload an image for classification.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with key `file` containing the image file

**Response (Success):**
```json
{
  "prediction": 0,
  "processed_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "image_size": "224x224"
}
```

**Response Fields:**
- `prediction`: Integer representing the predicted class (0, 1, 2, etc.)
- `processed_image`: Base64-encoded JPEG image (224×224 RGB) that was analyzed by the model
- `image_size`: Dimensions of the processed image

**Response (Error):**
```json
{
  "error": "error message"
}
```

## Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

### Using Python Requests

```python
import requests
import base64
from PIL import Image
import io

url = "http://localhost:8000/predict"
files = {"file": open("path/to/your/image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Image size: {result['image_size']}")

# Decode and display the processed image
img_data = result['processed_image'].split(',')[1]  # Remove data:image/jpeg;base64,
img_bytes = base64.b64decode(img_data)
processed_img = Image.open(io.BytesIO(img_bytes))
processed_img.show()  # Display the image
```

### Using JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Image size:', data.image_size);
    
    // Display the processed image
    const img = document.createElement('img');
    img.src = data.processed_image;
    document.body.appendChild(img);
  });
```

### Using the Medical Interface (HTML)

A ready-to-use medical interface is included as `medical_interface.html`. Simply:

1. **Start your API server:**
   ```powershell
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open `medical_interface.html` in your browser**

3. **Upload an image** and view:
   - The processed 224×224 RGB image
   - The predicted class
   - Class labels (customizable)

The interface includes:
- Drag-and-drop upload
- Real-time prediction
- Visual display of processed image
- Error handling
- Configurable API endpoint

## Deployment on Render

### Prerequisites
- A [Render](https://render.com/) account
- Git repository with your code

### Steps

1. **Push your code to a Git repository** (GitHub, GitLab, etc.)

2. **Create a new Web Service on Render:**
   - Go to your Render dashboard
   - Click "New +" and select "Web Service"
   - Connect your Git repository

3. **Configure the service:**
   - **Name:** `dfu-classification-api` (or your preferred name)
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port 10000`
   - **Plan:** Select your preferred plan (Free tier available)

4. **Deploy:**
   - Render will automatically detect the `render.yaml` file
   - Click "Create Web Service"
   - Wait for deployment to complete

5. **Access your API:**
   - Your API will be available at: `https://your-service-name.onrender.com`

### Important Notes for Render Deployment

- The free tier on Render may spin down after inactivity and take ~30 seconds to start up again
- Make sure your model file (`efficientnet_dfu_best.pth`) is included in your repository
- The model file is ~16MB, ensure your Git provider supports this file size
- Consider using Git LFS for large model files if needed

## Model Information

- **Architecture:** EfficientNet-B0
- **Input Size:** 224x224 RGB images
- **Output:** Integer class prediction
- **Best Validation Accuracy:** 98.73% (from training history)

## Image Processing Pipeline

1. Upload image via multipart/form-data
2. Open with PIL
3. Convert to RGB
4. Resize to 224×224
5. Convert to tensor with ImageNet normalization
6. Run inference without gradients
7. Return predicted class

## Error Handling

The API includes comprehensive error handling:
- Invalid image formats
- Model loading errors
- Inference errors
- File upload errors

All errors return HTTP 500 with a JSON response containing the error message.

## Development

### Adding New Features

The code is structured for easy extension:
- Model loading: `load_model()` function
- Prediction logic: `/predict` endpoint
- Additional preprocessing: Modify the `transform` pipeline

### Modifying Number of Classes

If your model has a different number of output classes, update line 36 in `app.py`:

```python
efficientnet_model.classifier[1] = nn.Linear(num_features, YOUR_NUM_CLASSES)
```

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, please check:
- FastAPI documentation: https://fastapi.tiangolo.com/
- PyTorch documentation: https://pytorch.org/docs/
- Render documentation: https://render.com/docs
