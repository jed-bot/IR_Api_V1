from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf  # Required for TFLite

app = FastAPI()

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="ingredient_classifier.tflite")
    interpreter.allocate_tensors()  # Allocate tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(" Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# Class labels (same as before)
class_labels = [
    'Bitter_m', 'Calamansi', 'Eggplant', 'Garlic', 'Ginger',
    'Okra', 'Onion', 'Pork', 'Potato', 'Squash', 'Tomato'
]

# Preprocess image (same as before)
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust size if needed
    img_array = np.array(img, dtype=np.float32)  # Use float32 for TFLite
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Predict using TFLite
def predict_ingredient(img: UploadFile):
    try:
        img_bytes = img.file.read()
        img = Image.open(BytesIO(img_bytes))
        processed_image = preprocess_image(img)

        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        result = [
            {"ingredient": class_labels[idx], "confidence": float(predictions[0][idx] * 100)}
            for idx in top_3_indices
        ]
        return {"predictions": result}

    except Exception as e:
        return {"error": str(e)}

# FastAPI endpoint (same as before)
@app.post("/predict/")
async def get_prediction(file: UploadFile = File(...)):
    return predict_ingredient(file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)