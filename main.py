from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import logging
from functools import lru_cache

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the model loading to prevent reloading on each request
@lru_cache(maxsize=1)
def load_ml_model():
    try:
        model = load_model("ingredient_classifier_200_epochs.h5")
        logger.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Ingredient labels
CLASS_LABELS = [
    'Bitter_m', 'Calamansi', 'Eggplant', 'Garlic', 'Ginger',
    'Okra', 'Onion', 'Pork', 'Potato', 'Squash', 'Tomato'
]

def preprocess_image(img):
    """Preprocess image for model prediction"""
    try:
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Ingredient Classifier API is running"}

@app.post("/predict/")
async def predict_ingredient(file: UploadFile = File(...)):
    """Predict ingredient from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load model (cached)
        model = load_ml_model()
        
        # Read and process image
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        processed_image = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_image)
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        # Format results
        results = [{
            "ingredient": CLASS_LABELS[idx],
            "confidence": float(predictions[0][idx] * 100)
        } for idx in top_3_indices]
        
        return JSONResponse(content={"predictions": results})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
