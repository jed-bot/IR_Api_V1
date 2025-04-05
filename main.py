from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
import matplotlib.pyplot as plt

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
try:
    model = load_model("ingredient_classifier_200_epochs.h5")  # Load the model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Ingredient labels (adjust these based on your dataset)
class_labels = [
    'Bitter_m', 'Calamansi', 'Eggplant', 'Garlic', 'Ginger',
    'Okra', 'Onion', 'Pork', 'Potato', 'Squash', 'Tomato'
]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model's expected input
    img_array = np.array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Function to predict the ingredient
def predict_ingredient(img: UploadFile):
    try:
        img_bytes = img.file.read()  # Read the image bytes
        img = Image.open(BytesIO(img_bytes))  # Open the image using PIL

        processed_image = preprocess_image(img)  # Preprocess the image
        predictions = model.predict(processed_image)  # Make the prediction

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]

        # Prepare result
        result = []
        for idx in top_3_indices:
            result.append({
                "ingredient": class_labels[idx],
                "confidence": float(predictions[0][idx] * 100)
            })

        return {"predictions": result}

    except Exception as e:
        return {"error": str(e)}

# Define an endpoint for predicting the ingredient from the image
@app.post("/predict/")
async def get_prediction(file: UploadFile = File(...)):
    return predict_ingredient(file)

# Run the app (use uvicorn to run it)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)